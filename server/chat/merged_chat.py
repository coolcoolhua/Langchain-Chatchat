from fastapi import Body, Request
from fastapi.responses import StreamingResponse,JSONResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                                  MERGED_MAX_DOCS_NUM)
from server.chat.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
import time

from webui_pages.utils import *
from configs.model_config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY
from langchain.utilities import BingSearchAPIWrapper
from langchain.docstore.document import Document


bad_words = [t.strip() for t in open('./server/chat/badwords.txt').readlines()]


def blocked_words_check(query):
    """遍历敏感词表

    Args:
        query (str): 用户输入的query

    Returns:
        should_be_blocked: Boolean, True代表需要被屏蔽
        searched_bad_word: str, 屏蔽词
    """    
    # 是否该被屏蔽
    should_be_blocked = False
    # 搜索到的屏蔽词
    searched_bad_word = ''
    for bad_word in bad_words:
        if bad_word in query:
            should_be_blocked = True
            searched_bad_word = bad_word
            break
    return should_be_blocked, searched_bad_word


def history_reformat(h) -> {}:
    """防止传入的history有问题，主要是针对UI交互的场景

    Returns:
        _type_: _description_
    """ 
    res = {
        "role":h.role, 
        "content":h.content
    }
    return res


def bing_search(text, result_len=SEARCH_ENGINE_TOP_K):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)


SEARCH_ENGINES = {"bing": bing_search,
                  }

def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
):
    results = SEARCH_ENGINES[search_engine_name](query, result_len=top_k)
    docs = search_result2docs(results)
    return docs

def kb_search_strategy(query,knowledge_base_name, top_k, score_shreshold):
    for i in range(3):
        docs = search_docs(query, knowledge_base_name, top_k, score_shreshold)
        if len(docs) < MERGED_MAX_DOCS_NUM:
            score_shreshold += 0.05
        print("docs",len(docs))
    return docs

def docs_merge_strategy(kb_docs, search_engine_docs, knowledge_base_name, request):
    final_docs = []
    source_documents = []
    if len(kb_docs) ==0:
        final_docs = search_engine_docs
        source_documents = [
            f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(search_engine_docs)
        ]
    else:
        final_docs.extend(kb_docs)
        for inum, doc in enumerate(kb_docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
            url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
        if len(kb_docs) < MERGED_MAX_DOCS_NUM:
            final_docs.extend(search_engine_docs[MERGED_MAX_DOCS_NUM - len(kb_docs):])
            source_documents.extend([
                f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
                for inum, doc in enumerate(search_engine_docs)
            ])
    return final_docs, source_documents
    
        
    

def merged_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                        score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),
                        history: List[History] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        ):
    
    ret = {
        "answer": "暂时无法回答该问题",
        "docs": ""
    }
    
    # 前处理，如果query里包含就直接结束
    check_res, blocked_word = blocked_words_check(query)
    if check_res == True:
        ret = {
            "answer": "该问题无法回答，因为问题中包含屏蔽词: "+ blocked_word,
            "docs" : "".join([])
        }
        return JSONResponse(ret)
    
    kb_docs = []
    searchengine_docs = []
    final_docs = []
    source_document = ""
    
    # kb搜索docs
    kb_docs = kb_search_strategy(query, knowledge_base_name, top_k, score_threshold)
    
    if len(kb_docs)<MERGED_MAX_DOCS_NUM:
        searchengine_docs = lookup_search_engine(query, "bing", top_k)
    
    final_docs, source_document = docs_merge_strategy(kb_docs, searchengine_docs,knowledge_base_name,request)
    
    print("最终docs",final_docs)
    print("最终source", source_document)
    
    context = "\n".join([doc.page_content for doc in final_docs])
    
    print("最终context",context)
    # 搜索引擎搜索docs

    api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)
    
    # 允许回答次数上限
    allowed_answer_times = 2
    
    temp_history = []
    if len(history)>0:
        if type(history[0]) == History:
            temp_history = [history_reformat(t) for t in history]
        else :
            temp_history = history
    else:
        temp_history = history
            
    print("now,history=",temp_history)
    
    
    for answered_time in range(1,allowed_answer_times+1):
        print("当前第",answered_time,'次回答')
        text = ""
        for d in api.docs_chat(query, knowledge_base_name, 5, 0.5, temp_history,final_docs,context):
            text += d["answer"]
        # for d in api.chat_chat(query,history):
        #     text +=d
        
        #后处理，如果回答中包含就重新生成
        check_res, blocked_word = blocked_words_check(text)
        
        if check_res == False:
            ret = {
                "answer" : text,
                "docs" : source_document
            }
            break
        else:
            if answered_time < allowed_answer_times:
                print("再次生成答案",answered_time)
                continue
            else:
                print("暂时无法回答该问题，因为该问题的回答中包含关键词: " + blocked_word)
                ret = {
                    "answer" : "暂时无法回答该问题，因为该问题的回答中包含关键词: " + blocked_word,
                    "docs" : "".join([])
                }
                
    
    print("最终返回",ret)              
    return JSONResponse(ret)