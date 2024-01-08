from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (LLM_MODEL)
from configs.kb_config import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
from configs import LLM_MODEL, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
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
from server.utils import get_prompt_template


def context_chat_tag(query: str = Body(..., description="用户输入", examples=["你好"]),
                        context: str = Body("",description="上下文"),
                        model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                        ):

    history = []
    print(context)
    print(query)
    async def knowledge_base_chat_iterator(query: str,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            callbacks=[callback],
        )
        
        prompt_template = get_prompt_template("search_tag")
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []

        answer = ""
        async for token in callback.aiter():
            answer += token
        yield json.dumps({"answer": answer,
                            "docs": source_documents},
                            ensure_ascii=False)

        await task

    return StreamingResponse(knowledge_base_chat_iterator(query, history, model_name),
                             media_type="text/event-stream")
