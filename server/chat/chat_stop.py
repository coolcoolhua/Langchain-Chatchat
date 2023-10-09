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
from langchain.output_parsers import ListOutputParser


stops = [t.strip() for t in open('./server/chat/badwords.txt').readlines()]


class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        for t in stops:
            if t in text:
                print(t)
                text = re.sub(t, "########", text)
        print(text)
        return text

    @property
    def _type(self) -> str:
        return "output_parser"
    
parser = CleanupOutputParser()

async def chat_stop(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                history: List[History] = Body([],
                                       description="历史对话",
                                       examples=[[
                                           {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                           {"role": "assistant", "content": "虎头虎脑"}]]
                                       ),
                stream: bool = Body(True, description="流式输出"),
                model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                prompt_name: str = Body("llm_chat", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
         ):
    history = [History.from_data(h) for h in history]
    print("history", history)
    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=[callback],
        )
        
        print("name",prompt_name)

        prompt_template = get_prompt_template(prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model, output_parser = parser)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query, "stop": stops}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield  parser.parse(token)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield parser.parse(answer)

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")

