from fastapi import Body, Request
from fastapi.responses import StreamingResponse,JSONResponse
from configs.model_config import (llm_model_dict, LLM_MODEL, PROMPT_TEMPLATE,
                                  VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD)
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


bad_words = [t.strip() for t in open('./server/chat/badwords.txt').readlines()]

api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)


bad_words = [t.strip() for t in open('./server/chat/badwords.txt').readlines()]

def blocked_words_check(query):
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



def kb_safe_chat_v2(query: str = Body(..., description="用户输入", examples=["你好"]),
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
    
    
    # 允许回答次数上限
    allowed_answer_times = 2
    
    
    # 前处理，如果query里包含就直接结束
    check_res, blocked_word = blocked_words_check(query)
    if check_res == True:
        ret = {
            "answer": "该问题无法回答，因为问题中包含屏蔽词: "+ blocked_word
        }
        return JSONResponse(ret)
    
    
    for answered_time in range(1,allowed_answer_times+1):
        print("当前第",answered_time,'次回答')
        text = ""
        docs = ""
        for d in api.knowledge_base_chat(query, "samples", 5, 0.5, history):
            text += d["answer"]
            docs = "\n\n".join(d["docs"])
        # for d in api.chat_chat(query,history):
        #     text +=d
        print(text)
        
        #后处理，如果回答中包含就重新生成
        check_res, blocked_word = blocked_words_check(text)
        
        if check_res == False:
            ret = {
                "answer" : text,
                "docs" : docs
            }
            print(ret)
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
                  
    return JSONResponse(ret)