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


role_definition = """
角色：
生涯辅导老师

背景：
中国的高中生们普遍缺乏对于职业生涯发展的探索。你作为知名的生涯辅导老师，有义务和能力改变他们的认知，让他们以轻松，非常深入浅出的方式获取各种职业学科的相关知识。

目标：
以专业且善解人意的态度，让高中生对了解一个职业或学科

技能：
1、对各种职业和专业都了如指掌。
2、善于分析学生的问题是需要提供事实的“事实型问题”，还是需要提供观点的“观点型问题”。
事实型问题（比如：“心理学的名人有谁”，“什么是心理学”）
观点型问题（比如：“我是否适合学习心理学”）
3、擅长从提供的已知信息提取答案。

限制：
回答尽量详细，不要出现“指令”，“已知信息”等内容。

回答流程：
1、问题类型判断：判断问题是事实型问题还是观点型问题
2、输出答案：
对于事实型问题，优先从已知信息提取答案。如果无法从中得到答案，忽略已知内容直接回答问题。
对于观点型问题，以客观中立的态度，在回答中以“正面”和”反面“两个方面进行回答。
3、总结
对回答的答案附上总结。

"""



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
        # print("当前使用分值",score_shreshold)
        docs = search_docs(query, knowledge_base_name, top_k, score_shreshold)
        # print("最终文档",docs)
        if len(docs) < MERGED_MAX_DOCS_NUM:
            score_shreshold += 0.05
        else:
            break
    return docs

def docs_merge_strategy(kb_docs, search_engine_docs, knowledge_base_name, request):
    final_docs = []
    source_documents = []
    if len(kb_docs) ==0:
        final_docs = search_engine_docs
        source_documents = [
            f"""##### 搜索出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n""".replace('@@@@@@@@@@','') + '\n----\n'
            for inum, doc in enumerate(search_engine_docs)
        ]
    else:
        final_docs.extend(kb_docs)
        for inum, doc in enumerate(kb_docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
            url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""##### 知识库出处 [{inum + 1}] [{filename.replace('.txt','')}] \n\n匹配分数{str(1-doc.score)[:5]} \n\n{doc.page_content}\n\n""".replace('@@@@@@@@@@','') + '\n----\n'
            source_documents.append(text)
        if len(kb_docs) < MERGED_MAX_DOCS_NUM:
            supplement_num = MERGED_MAX_DOCS_NUM - len(kb_docs)
            search_engine_docs = search_engine_docs[:supplement_num]
            final_docs.extend(search_engine_docs)
            source_documents.extend([
                f"""##### 搜索出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n""".replace('@@@@@@@@@@','') + '\n----\n'
                for inum, doc in enumerate(search_engine_docs)
            ])
    return final_docs, source_documents
    
        
    

def merged_chat_diytemplate(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                        used_template: str = Body(..., description="使用的template", examples=["你是一个生涯辅导老师"]),
                        top_k: int = Body(MERGED_MAX_DOCS_NUM, description="最大匹配向量数"),
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
                        request: Request = None
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
    print("知识库共n篇",len(kb_docs))
    
    
    # 开启/关闭搜索文章
    # if len(kb_docs)<MERGED_MAX_DOCS_NUM:
    #     searchengine_docs = lookup_search_engine(query, "bing", top_k)
    
    final_docs, source_document = docs_merge_strategy(kb_docs, searchengine_docs,knowledge_base_name,request)
    
    # print("最终docs",final_docs)
    # print("最终source", source_document)
    
    final_docs.reverse()
    
    context = "\n".join([doc.page_content for doc in final_docs]).replace('@@@@@@@@@@\n','')
    
    # print("最终context",context)
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
    
    
    # 增加角色定义
    # prefix_history = [{
    #     "role": "system",
    #     "content": role_definition
    # }]
    
    # if prefix_history[0] not in temp_history:
    #     temp_history = prefix_history + temp_history
        
    
    
    
    for answered_time in range(1,allowed_answer_times+1):
        print("当前第",answered_time,'次回答')
        text = ""
        for d in api.docs_chat_diytemplate(query, knowledge_base_name, used_template, 5, score_threshold, temp_history,final_docs,context):
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
                print("暂时无法回答该问题，因为该问题的回答中包含关键词: " + blocked_word + '。被过滤前的答案为:\n\n' + text)
                ret = {
                    "answer" : "暂时无法回答该问题，因为该问题的回答中包含关键词: " + blocked_word + '。被过滤前的答案为:\n\n' + text,
                    "docs" : "".join([])
                }
                           
    return JSONResponse(ret)