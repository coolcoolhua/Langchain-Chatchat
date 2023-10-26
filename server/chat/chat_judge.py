from fastapi import Body
from fastapi.responses import StreamingResponse
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

def chat_judge(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
         history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
         stream: bool = Body(False, description="流式输出"),
         model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
         ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = "llm_chat",
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            callbacks=[callback],
        )

        prompt_template = get_prompt_template("llm_chat")
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        print("chat judge最终prompt", chat_prompt)
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name="llm_chat"),
                             media_type="text/event-stream")
