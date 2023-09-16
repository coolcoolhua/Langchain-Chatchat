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
高中生涯辅导老师，为高中生做生涯探索辅导。你主要回答心理学方面的问题。
背景：
中国的高中生们普遍缺乏对于职业生涯发展的探索。你作为知名的生涯辅导老师，有义务和能力改变他们的认知，让他们以轻松，非常深入浅出的方式获取各种职业学科的相关知识。
目标：
1、以专业且善解人意的态度，让高中生对了解一个职业或学科。
2、每个学生的信息都不相同，如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答。
限制：
用亲切的语气回答，回答尽量详细。
不要出现“指令”，“已知信息”等内容。
不要描述自己的语言风格等内容。

"""

judge_template = """判断以下问题是「闲聊问题」还是和生涯教育相关的「观点问题」还是「事实问题」。
示例问题：“你好”，示例答案：“「闲聊问题」”。
示例问题：“你是谁”，示例答案：“「闲聊问题」”。
示例问题：“我适合学心理学吗”，示例答案：“「观点问题」”。
示例问题：“什么是心理学”，示例答案：“「事实问题」”。
注意：和心理学相关的问题都不是闲聊问题！
问题：“{}”, 答案:
"""

truth_template = """<角色>你是一个高中生涯教育老师，你主要回答心理学方面的问题。</角色>
<指令>优先从已知信息提取答案。
如果无法从中得到答案，忽略已知内容，根据回答历史和上下文，直接回答问题。
回答尽量详细，不要出现“角色”，“指令”，“已知信息”内的内容。</指令>
<历史信息>{{ history }}</历史信息>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>                 
"""

opinion_template = """<角色>你是一个高中生涯教育老师，你主要回答心理学方面的问题。</角色>
<指令>优先从已知信息提取答案。
以客观中立的态度，在回答中以“正面”和”反面“两个方面进行回答，最后附上总结。
如果无法从中得到答案，忽略已知内容，根据回答历史和上下文，直接回答问题。
回答尽量详细，不要出现“角色”，“指令”，“已知信息”内的内容。</指令>
<历史信息>{{ history }}</历史信息>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>
"""

chat_template = """<角色>你是一个高中生涯教育老师，你主要回答心理学方面的问题。</角色>
<限制>只能说自己是个高中生涯教育老师。
不要描述自己。
</限制>
<问题>{}</问题>
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
    
    
def question_type_judge(text,docs_len):
    if '闲聊' in text or docs_len == 0:
        return '闲聊'
    elif '事实' in text:
        return '事实'
    else:
        return '观点'
        
    

def merged_chat_v2(query: str = Body(..., description="用户输入", examples=["你好"]),
                        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
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
    
    # 允许回答次数上限
    allowed_answer_times = 2
    # 知识库文档合集
    kb_docs = []
    # 搜索引擎文档合集
    searchengine_docs = []
    # 最终文档合集
    final_docs = []
    # 展示用文档markdown
    source_document = ""
    
    # 返回模版
    ret = {
        "answer": "暂时无法回答该问题",
        "docs": "",
        "question_type": ""
    }
    
    # 前处理，如果query里包含就直接结束
    check_res, blocked_word = blocked_words_check(query)
    if check_res:
        ret["answer"]="该问题无法回答，因为问题中包含屏蔽词: "+ blocked_word
        return JSONResponse(ret)
    

    api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)

    
    final_history = []
    if len(history)>0:
        if type(history[0]) == History:
            final_history = [history_reformat(t) for t in history]
        else :
            final_history = history
    else:
        final_history = history
            
    
    # 增加角色定义
    prefix_history = [{
        "role": "system",
        "content": role_definition
    }]
    
    if prefix_history[0] not in final_history:
        final_history = prefix_history + final_history
    
    # step1 判断是哪种类型的问题
    judge_prompt = judge_template.format(query)
    r = api.chat_judge(judge_prompt, history=final_history)
    judge_text = ""
    for t in r:
        judge_text += t

    docs_len = len(search_docs(query, knowledge_base_name, top_k, score_threshold))
    question_type = question_type_judge(judge_text,docs_len)
    
    print("类别判断为",question_type)
    
    if question_type !='闲聊':
        # kb搜索docs
        # 如果query里不包含专业名，自动加上
        kb_query = query if knowledge_base_name in query else knowledge_base_name + "的" + query
        print("知识库搜索query",kb_query)
        kb_docs = kb_search_strategy(kb_query, knowledge_base_name, top_k, score_threshold)
        print("知识库共n篇",len(kb_docs))
        
        # 搜索引擎搜索docs
        # 开启/关闭搜索文章
        if len(kb_docs)<MERGED_MAX_DOCS_NUM:
            searchengine_docs = lookup_search_engine(kb_query, "bing", top_k)
        
        print("搜索库共n篇",len(searchengine_docs))
        
        final_docs, source_document = docs_merge_strategy(kb_docs, searchengine_docs,knowledge_base_name,request)
        
        # print("最终docs",final_docs)
        # print("最终source", source_document)
        
        # 逆反搜索结果，越重要的越靠近问题
        final_docs.reverse()
        
        # 模型最终看到的上下文
        context = "\n".join([doc.page_content for doc in final_docs]).replace('@@@@@@@@@@\n','')
        # print("最终context",context)
    

    for answered_time in range(1,allowed_answer_times+1):
        print("当前第",answered_time,'次回答')
        
        text = ""
        if question_type ==  "闲聊":
            chat_prompt = chat_template.format(query)
            for d in api.chat_judge(chat_prompt, history=final_history):
                text += d
            source_document = ""
        else:
            if question_type == '事实':
                used_template = truth_template
            else:
                used_template = opinion_template
                
                
            for d in api.docs_chat_diytemplate(query, knowledge_base_name, used_template, 5, score_threshold, final_history,final_docs,context):
                text += d["answer"]
        
        
        #后处理，如果回答中包含就重新生成
        check_res, blocked_word = blocked_words_check(text)
        if check_res == False:
            ret = {
                "answer" : text,
                "docs" : source_document,
                "question_type": question_type
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
                    "docs" : "",
                    "question_type": question_type
                }
                           
    return JSONResponse(ret)