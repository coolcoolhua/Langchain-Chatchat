from __future__ import annotations
from fastapi import Body, Request
from fastapi.responses import StreamingResponse, JSONResponse
from configs.model_config import LLM_MODEL, TEMPERATURE
from configs.kb_config import (
    SCORE_THRESHOLD,
    VECTOR_SEARCH_TOP_K,
    MERGED_MAX_DOCS_NUM,
    BING_SEARCH_URL,
    BING_SUBSCRIPTION_KEY,
)

from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse
from langchain.chat_models import ChatOpenAI
from langchain.llms.utils import enforce_stop_tokens
from langchain.chains import LLMChain
# from langchain.callbacks import AsyncIteratorCallbackHandler
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
import re
from langchain.schema import BaseOutputParser

from webui_pages.utils import *
from langchain.utilities import BingSearchAPIWrapper
from langchain.docstore.document import Document


bad_words = [t.strip() for t in open("./server/chat/badwords.txt").readlines()]
import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal, Union, cast

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema.output import LLMResult

# TODO If used by two LLM runs in parallel this won't work as expected


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):

    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.llm_status = 0
        self.generate_length  = 0
        self.t0 = time.time()
        self.t1 = ''

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        print("llm start")
        self.generate_length = 0
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # print("生成token",token)
        if token is not None and token != "" and token not in bad_words:
            self.generate_length += len(token)
            self.queue.put_nowait(token)
        else:
            if len(token) > 0 and token in bad_words:
                self.llm_status = 1
                print("存在敏感词",token,"修改llmstatus",self.llm_status)
                # self.queue.put_nowait("########")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()
        self.t1 = time.time()
        print('大模型共生成：', self.generate_length , 'token','，花费时间', round(self.t1 - self.t0, 2), 's', '生成速度',self.generate_length/(self.t1 - self.t0),'token/s')
        print("llm finished")

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.done.set()
        print("llm error")

    # TODO implement the other methods

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            token_or_done = cast(Union[str, Literal[True]], done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if token_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield token_or_done



class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        for t in bad_words:
            if t in text:
                print(t)
                # text = re.sub(t, "########", text)
        # print(text)
        return text

    @property
    def _type(self) -> str:
        return "output_parser"


parser = CleanupOutputParser()


role_definition = """
角色：
高中生涯辅导老师，为高中生做生涯探索辅导。可以为学生提供各个学科，专业方向的指导。
背景：
中国的高中生们普遍缺乏对于职业生涯发展的探索。你作为知名的生涯辅导老师，有义务和能力改变他们的认知，让他们以轻松，非常深入浅出的方式获取各种职业学科的相关知识。
目标：
1、以专业且善解人意的态度，让高中生对了解一个职业或学科。
2、每个学生的信息都不相同，如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答。
限制：
用亲切的语气回答，回答尽量详细。
不要出现“指令”，“已知信息”等内容。
不要描述自己的语言风格等内容。
注意：
已知信息里出现的内容都是第三方资料，不是当前学生的资料。
"""


truth_template = """<角色>你是一个高中生涯教育老师，不允许说自己是大语言模型/角色>
<指令>优先从已知信息提取答案。
如果无法从中得到答案，忽略已知内容，根据回答历史和上下文，直接回答问题。
回答尽量详细，不要出现“角色”，“指令”，“已知信息”内的内容。</指令>
<历史信息>{{ history }}</历史信息>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>      
注意：已知信息的内容里的内容不是当前咨询者的信息，是第三方资料。           
"""

opinion_template = """<角色>你是一个高中生涯教育老师，不允许说自己是大语言模型</角色>
<指令>优先从已知信息提取答案。
回答尽量详细，要以客观中立的态度，在回答中以“正面”和”反面“两个方面进行回答，最后附上总结。
如果无法从中得到答案，忽略已知内容，根据回答历史和上下文，直接回答问题。
，不要出现“角色”，“指令”，“已知信息”内的内容。如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答</指令>
<历史信息>{{ history }}</历史信息>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>
注意：已知信息的内容里的内容不是当前咨询者的信息，是第三方资料。
"""

chat_template = """<角色>你是一个高中生涯教育老师，可以为学生提供各个学科，专业方向的指导。不允许说自己是大语言模型</角色>
<限制>只能说自己是个高中生涯教育老师。不要描述自己。如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答</限制>
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
    searched_bad_word = ""
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
    res = {"role": h.role, "content": h.content}
    return res


def bing_search(text, result_len=SEARCH_ENGINE_TOP_K):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [
            {
                "snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                "title": "env info is not found",
                "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html",
            }
        ]
    search = BingSearchAPIWrapper(
        bing_subscription_key=BING_SUBSCRIPTION_KEY, bing_search_url=BING_SEARCH_URL
    )
    return search.results(text, result_len)


SEARCH_ENGINES = {
    "bing": bing_search,
}


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"] if "snippet" in result.keys() else "",
            metadata={
                "source": result["link"] if "link" in result.keys() else "",
                "filename": result["title"] if "title" in result.keys() else "",
            },
        )
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


def kb_search_strategy(query, knowledge_base_name, top_k, score_shreshold):
    for i in range(1):
        # print("当前使用分值",score_shreshold)
        docs = search_docs(query, knowledge_base_name, top_k, score_shreshold)
        print("搜索条件太严苛，申请加分")
        # print("最终文档",docs)
        if len(docs) == 0:
            score_shreshold += 0.05
        else:
            break
    return docs


def docs_merge_strategy(kb_docs, search_engine_docs, knowledge_base_name, request):
    final_docs = []
    source_documents = []
    if len(kb_docs) == 0:
        final_docs = search_engine_docs
        source_documents = [
            f"""##### 搜索出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n搜索标题 {doc.metadata["filename"].replace('<b>','').replace('</b>','')}""".replace(
                "@@@@@@@@@@", ""
            )
            for inum, doc in enumerate(search_engine_docs)
        ]
    else:
        final_docs.extend(kb_docs)
        for inum, doc in enumerate(kb_docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            parameters = urlencode(
                {"knowledge_base_name": knowledge_base_name, "file_name": filename}
            )
            url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = (
                f"""##### 知识库出处 [{inum + 1}] [{filename.replace('.txt','')}] \n\n匹配分数{str(1-doc.score)[:5]} \n\n{doc.page_content}\n\n""".replace(
                    "@@@@@@@@@@", ""
                )
                
            )
            source_documents.append(text)
        if len(kb_docs) < MERGED_MAX_DOCS_NUM:
            supplement_num = MERGED_MAX_DOCS_NUM - len(kb_docs)
            search_engine_docs = search_engine_docs[:supplement_num]
            final_docs.extend(search_engine_docs)
            source_documents.extend(
                [
                    f"""##### 搜索出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n搜索标题 {doc.metadata["filename"].replace('<b>','').replace('</b>','')}""".replace(
                        "@@@@@@@@@@", ""
                    )
                    
                    for inum, doc in enumerate(search_engine_docs)
                ]
            )
    return final_docs, source_documents


def question_type_judge(text, docs_len):
    print("模型对query的判断是", text)
    # 原始query既搜不到 也被判定成闲聊，才算闲聊
    if "闲聊" in text and docs_len == 0:
        return "闲聊"
    else:
        return "相关"


# bert判断问题是否是闲聊
def get_idle_res(query):
    import requests

    url = "http://127.0.0.1:7861/chat/bert_chat_judge"
    payload = {"query": query, "test": "sdf"}
    response = requests.request("POST", url, data=json.dumps(payload))
    return json.loads(response.text)["answer"]


# bert判断问题是否是事实或观点
def get_truth_res(query):
    import requests

    url = "http://127.0.0.1:7861/chat/bert_truth_judge"
    payload = {"query": query, "test": "sdf"}
    response = requests.request("POST", url, data=json.dumps(payload))
    return json.loads(response.text)["answer"]


def career_flow_chat(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
    top_k: int = Body(MERGED_MAX_DOCS_NUM, description="最大匹配向量数"),
    score_threshold: float = Body(
        SCORE_THRESHOLD,
        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
        ge=0,
        le=1,
    ),
    history: List[History] = Body(
        [],
        description="历史对话",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Body(True, description="流式输出"),
    local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
    request: Request = None,
):
    # 计时
    t0 = time.time()

    # 知识库文档合集
    kb_docs = []
    # 搜索引擎文档合集
    searchengine_docs = []
    # 最终文档合集
    final_docs = []
    # 展示用文档markdown
    source_document = ""

    # 返回模版
    ret = {"answer": "暂时无法回答该问题", "docs": "", "question_type": ""}

    # 前处理，如果query里包含就直接结束
    check_res, blocked_word = blocked_words_check(query)
    if check_res:
        ret["answer"] = "该问题无法回答，因为问题中包含屏蔽词: " + blocked_word
        return JSONResponse(ret)

    api = ApiRequest(base_url="http://127.0.0.1:7861", no_remote_api=False)

    final_history = []
    if len(history) > 0:
        if type(history[0]) == History:
            final_history = [history_reformat(t) for t in history]
        else:
            final_history = history
    else:
        final_history = history

    # 增加角色定义
    prefix_history = [{"role": "system", "content": role_definition}]

    if prefix_history[0] not in final_history:
        final_history = prefix_history + final_history

    # step1 判断是哪种类型的问题
    judge_text = get_idle_res(query)
    print("bert模型回答", judge_text)
    t1 = time.time()
    print("大模型意图判断响应耗时为：", round(t1 - t0, 2), "s", "总共时间", round(t1 - t0, 2), "s")

    # step2 在知识库里搜索
    kb_docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
    docs_len = len(kb_docs)
    print("知识库搜索query", query)
    print("原始query在知识库的搜索结果共n篇", docs_len)

    t2 = time.time()
    print("知识库搜索耗时为：", round(t2 - t1, 2), "s", "总共时间", round(t2 - t0, 2), "s")

    question_type = question_type_judge(judge_text, docs_len)
    print("生涯相关性判断为,", question_type)
    print("第一阶段响应耗时为：", round(time.time() - t0, 2), "s")

    t_extra = ""

    if question_type != "闲聊":
        question_type = get_truth_res(query)
        print("最终判断问题类型为", question_type)
        t_extra = time.time()
        print(
            "事实观点判断耗时为：",
            round(t_extra - t2, 2),
            "s",
            "总共时间",
            round(t_extra - t0, 2),
            "s",
        )

        # kb搜索docs
        # 如果query里不包含专业名，自动加上
        # kb_docs = kb_search_strategy(kb_query, knowledge_base_name, top_k, score_threshold)

        # 搜索引擎搜索docs
        search_query = (
            query if knowledge_base_name in query else knowledge_base_name + "的" + query
        )
        if len(kb_docs) < MERGED_MAX_DOCS_NUM:
            try:
                searchengine_docs = lookup_search_engine(
                    search_query, "bing", MERGED_MAX_DOCS_NUM - docs_len
                )
            except:
                searchengine_docs = []
            print("搜索库共n篇", len(searchengine_docs))

        final_docs, source_documents = docs_merge_strategy(
            kb_docs, searchengine_docs, knowledge_base_name, request
        )

        # 逆反搜索结果，越重要的越靠近问题
        final_docs.reverse()

        # 模型最终看到的上下文
        divide_length =  int(4096/MERGED_MAX_DOCS_NUM)
        context = "\n".join([doc.page_content[:divide_length] for doc in final_docs]).replace(
            "@@@@@@@@@@\n", ""
        )
        
        # 如果context长度过长，只取前4096，因为是baichuan的限制
        if len(context)>=4096:
            context = context[:4096]
        
        # print("最终context",context)

    t3 = time.time()
    if len(searchengine_docs) > 0:
        print("搜索引擎搜索耗时为：", round(t3 - t_extra, 2), "s", "总共时间", round(t3 - t0, 2), "s")

    print("前处理流程结束，共耗时", round(t3 - t0, 2), "s")

    text = ""
    if question_type == "闲聊":
        used_template = chat_template.format(query)
        context = ""
        source_documents = ""
    else:
        if question_type == "事实":
            used_template = truth_template
        else:
            used_template = opinion_template

    async def knowledge_base_chat_iterator(
        query: str, history: Optional[List[History]], model_name: str = LLM_MODEL
    ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        print("使用模型",model_name,"模型温度",TEMPERATURE)
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            callbacks=[callback],
        )

        input_msg = History(role="user", content=used_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg]
        )
        print("最终prompt", chat_prompt)
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(
            wrap_done(
                chain.acall(
                    # {"context": context, "question": query, "history": history, "stop":bad_words}
                    {"context": context, "question": query, "history": history}
                ),
                callback.done,
            ),
        )

        # print("kb最终source",source_documents)
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": parser.parse(token), "llm_status": callback.llm_status}, ensure_ascii=False)
                # yield json.dumps({"answer": enforce_stop_tokens(token,bad_words)}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += parser.parse(token)
                # answer += enforce_stop_tokens(token,bad_words)
            yield json.dumps(
                {"answer": answer, "docs": source_documents}, ensure_ascii=False
            )

        await task

    return StreamingResponse(
        knowledge_base_chat_iterator(
            query=query, history=history, model_name=LLM_MODEL
        ),
        media_type="text/event-stream",
    )
