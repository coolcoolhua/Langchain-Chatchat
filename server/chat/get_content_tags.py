from fastapi import Body
from fastapi.responses import StreamingResponse, JSONResponse
from configs import LLM_MODEL, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History
from langchain.schema import BaseOutputParser
import re
from server.utils import get_prompt_template
from webui_pages.utils import *


def get_content_tags(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
                stream: bool = Body(False, description="流式输出"),
                model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                prompt_name: str = Body("llm_chat", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                content: str = Body(..., description="文章内容"),
         ):
    history = []
    api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)
        
    # 返回模版
    ret = {
        "answer": ""
    }
    text = ""
    for d in api.get_content_tags_stream(query = query, history = [], stream = False, model = LLM_MODEL, temperature=1, prompt_name = "ss", content=content):
        print(d)
        text += d
        
    ret["answer"] = text

    return JSONResponse(ret)

