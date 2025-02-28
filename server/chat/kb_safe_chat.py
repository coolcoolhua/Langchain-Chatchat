from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs.model_config import (LLM_MODEL)
from configs.kb_config import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
from server.utils import wrap_done
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


bad_words = [t.strip() for t in open('./server/chat/badwords.txt').readlines()]



def kb_safe_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
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
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def kb_safe_chat_iterator(query: str,
                                           kb: KBService,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           ) -> AsyncIterable[str]:
      
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])
        prompt_template = """"""
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        # Begin a task that runs in the background.
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
            
        answered_times = 0
        word_safe = True
        allowd_answered_times = 2
        
        for index in range(allowd_answered_times):
            print('第',index,'尝试回答')
            
            callback = AsyncIteratorCallbackHandler()

            model = ChatOpenAI(
                streaming=True,
                verbose=True,
                callbacks=[callback],
                openai_api_key=llm_model_dict[LLM_MODEL]["api_key"],
                openai_api_base=llm_model_dict[LLM_MODEL]["api_base_url"],
                model_name=LLM_MODEL,
                openai_proxy=llm_model_dict[LLM_MODEL].get("openai_proxy")
            )
            chain = LLMChain(prompt=chat_prompt, llm=model)

            # Begin a task that runs in the background.
            
            
            if answered_times >0:
                query += '请不要在回答里包含'+bad_word
            print(query)
            
            task = asyncio.create_task(wrap_done(
                chain.acall({"context": context, "question": query}),
                callback.done),
            )
                
            if stream:
                async for token in callback.aiter():
                    # Use server-sent-events to stream the response
                    yield json.dumps({"answer": token,
                                  "docs": source_documents},
                                 ensure_ascii=False)
            else:
                answer = ""
                start = time.time()
                
                async for token in callback.aiter():
                    answer += token
                    
                print('答案长度',len(answer))
                end = time.time()
                elapsed = end - start
                print('生成答案耗时: '+ time.strftime("%H:%M:%S", time.gmtime(elapsed)))
                    
                word_safe = True
                bad_word = '无'
                for t in bad_words:
                    if t in answer:
                        word_safe = False
                        bad_word = t 
                        break
                end1 = time.time()
                print(end1,end)
                elapsed1 = end1 - end
                print('遍历词表耗时: '+ time.strftime("%H:%M:%S", time.gmtime(elapsed)))
                answer = '安全回答判断\n\n是否存在禁用词 = ' + str(not word_safe) + '\n\n' + '禁用词 = ' + bad_word + '\n\n' + '答案长度 =  '+str(len(answer))+ '\n\n' + '生成答案耗时= '+ str(time.strftime("%H:%M:%S", time.gmtime(elapsed)))+ '\n\n' + '遍历词表耗时: '+ str(time.strftime("%H:%M:%S", time.gmtime(elapsed1))) + '\n\n' + '实际答案 : ' + answer
                answered_times +=1
                    
                if word_safe == True:
                    print(answer)
                    yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
                    break
                else:
                    if answered_times < allowd_answered_times:
                        print('本次生成答案',answer)
                        print('再做一次尝试')
                        answered_times += 1
                        continue
                    else:
                        print('本次生成答案',answer)
                        answer = '暂时无法回答这个问题，因为生成的答案包含屏蔽词:' + bad_word
                        yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
            await task

      
    
    return StreamingResponse(kb_safe_chat_iterator(query, kb, top_k, history),
                             media_type="text/event-stream")
