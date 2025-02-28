from langchain.utilities import BingSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
from configs.kb_config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY, SEARCH_ENGINE_TOP_K
from fastapi import Body
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.concurrency import run_in_threadpool
from configs.model_config import (LLM_MODEL, TEMPERATURE)
from server.utils import wrap_done
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
from server.chat.utils import History
from langchain.docstore.document import Document
import json
from webui_pages.utils import *

def bing_search(text, result_len=SEARCH_ENGINE_TOP_K):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)

SEARCH_ENGINES = {"bing": bing_search}


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
    search_engine = SEARCH_ENGINES[search_engine_name]
    results = search_engine(query, result_len=top_k)
    print(results)
    docs = search_result2docs(results)
    return docs


def unsatisfy_question_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                            top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                            history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                "content": "虎头虎脑"}]]
                                                            ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", gt=0.0, le=1.0),
                       ):
    
    search_engine_name = 'bing'
    stream = False

    history = [History.from_data(h) for h in history]

    docs = lookup_search_engine(query, search_engine_name, top_k)
    context = "\n".join([doc.page_content for doc in docs])

    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(docs)
    ]
    
    text = ""
    api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)
    for d in api.search_engine_chat(
                query,
                'bing',
                3,
                model=LLM_MODEL,
                temperature=TEMPERATURE,
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
    ret = {
        "answer": text,
        "docs": source_documents
    }
    
    return JSONResponse(ret)
