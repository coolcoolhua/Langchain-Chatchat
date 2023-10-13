import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs import (LLM_MODEL, MERGED_MAX_DOCS_NUM, TEMPERATURE)

from server.utils import get_model_worker_config
from typing import List, Dict
import pypinyin

def to_pinyin_key(s):  
   return pypinyin.pinyin(s[0])[0]

chat_box = ChatBox(
    assistant_avatar=os.path.join("img", "chatchat_icon_blue_square_v2.png")
)

judge_template = """判断以下问题与职业生涯教育或{}是否相关，答案可选项「相关」，「不相关」。
示例问题：“你好”，示例答案：“「不相关」”。
示例问题：“你是谁”，示例答案：“「不相关」”。
示例问题：“我适合学心理学吗”，示例答案：“「相关」”。
示例问题：“什么是金融学”，示例答案：“「相关」”。
根据示例答案直接给出答案，只允许出现答案可选项中的内容。
问题：“{}”, 答案:
"""

opinion_truth_judge_template = """
判断以下问题是事实性问题还是观点性问题。
示例问题：“什么是计算机科学”，示例答案：“「事实性问题」”。
示例问题：“计算机科学排名”，示例答案：“「事实性问题」”。
示例问题：“我适合学金融学吗”，示例答案：“「观点性问题」”。
示例问题：“计算机专业数学难吗”，示例答案：“「观点性问题」”。
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

opinion_template = """<角色>你是一个高中生涯教育老师</角色>
<指令>优先从已知信息提取答案。
回答尽量详细，要以客观中立的态度，在回答中以“正面”和”反面“两个方面进行回答，最后附上总结。
如果无法从中得到答案，忽略已知内容，根据回答历史和上下文，直接回答问题。
，不要出现“角色”，“指令”，“已知信息”内的内容。如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答</指令>
<历史信息>{{ history }}</历史信息>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>
"""

chat_template = """<角色>你是一个高中生涯教育老师，可以为学生提供各个学科，专业方向的指导</角色>
<限制>只能说自己是个高中生涯教育老师。不要描述自己。如果回答的问题需要学生的信息(比如省份，成绩等)，发问让他回答</限制>
<问题>{}</问题>
"""


unsatisfy_prompt = ""


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


def dialogue_page(api: ApiRequest):
    chat_box.init_session()

    with st.sidebar:
        # TODO: 对话模型与会话绑定
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        dialogue_mode = st.selectbox(
            "请选择对话模式",
            ["流式问答","融合问答", "搜索引擎问答", "LLM 对话", "知识库问答"],
            on_change=on_mode_change,
            key="dialogue_mode",
        )

        def on_llm_change():
            config = get_model_worker_config(llm_model)
            if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                st.session_state["prev_llm_model"] = llm_model
            st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        def reanswer1():
            text = ""
            for dt in api.search_engine_chat(
                unsatisfy_prompt, "bing", 3, model=llm_model
            ):
                if error_msg := check_error_msg(dt):  # check whether error occured
                    st.error(error_msg)
                else:
                    text += dt["answer"]
            chat_box.update_msg("\n\n".join(dt["docs"]), 1, streaming=False)
            chat_box.update_msg(text, 0, streaming=False)

        def save_prompt(prompt, process_name):
            with open("good_prompt_" + process_name + ".txt", "a") as f:
                f.write(prompt + "\n\n\n")
                st.toast("prompt保存成功")

        # judge_template_save = st.text_area(
        #     "判断问题类型 使用的prompt是", judge_template, height=200
        # )
        # if st.button("这个prompt很棒，保存一下", key="judge", use_container_width=True):
        #     save_prompt(judge_template_save, "判断")
        # opinion_truth_judge_template_save = st.text_area(
        #     "判断问题类型 使用的prompt是", opinion_truth_judge_template, height=200
        # )
        # if st.button("这个prompt很棒，保存一下", key="truth_opinion_judge", use_container_width=True):
        #     save_prompt(opinion_truth_judge_template, "判断")

        # chat_template_save = st.text_area("闲聊问题 使用的prompt是", chat_template, height=200)
        # if st.button("这个prompt很棒，保存一下", key="chat", use_container_width=True):
        #     save_prompt(judge_template_save, "闲聊")
        # truth_template_save = st.text_area(
        #     "事实问题 使用的prompt是", truth_template, height=200
        # )
        # if st.button("这个prompt很棒，保存一下", key="truth", use_container_width=True):
        #     save_prompt(truth_template_save, "事实")
        # opinion_template_save = st.text_area(
        #     "观点问题 使用的prompt是", opinion_template, height=200
        # )
        # if st.button("这个prompt很棒，保存一下", key="opinion", use_container_width=True):
        #     save_prompt(opinion_template_save, "观点")

        running_models = api.list_running_models()
        available_models = []
        config_models = api.list_config_models()
        for models in config_models.values():
            for m in models:
                if m not in running_models:
                    available_models.append(m)
        llm_models = running_models + available_models
        index = llm_models.index(st.session_state.get("cur_llm_model", LLM_MODEL))
        llm_model = st.selectbox("选择LLM模型：",
                                 llm_models,
                                 index,
                                 format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model",
                                 )
        if (st.session_state.get("prev_llm_model") != llm_model
                and not get_model_worker_config(llm_model).get("online_api")
                and llm_model not in running_models):
            with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.01)

        ## 部分模型可以超过10抡对话
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

        if dialogue_mode == "知识库问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)

                ## Bge 模型会超过1
                score_threshold = st.slider("知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)

                # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)
        elif dialogue_mode == "搜索引擎问答":
            search_engine_list = list(SEARCH_ENGINES.keys())
            with st.expander("搜索引擎配置", True):
                search_engine = st.selectbox(
                    label="请选择搜索引擎",
                    options=search_engine_list,
                    index=search_engine_list.index("duckduckgo")
                    if "duckduckgo" in search_engine_list
                    else 0,
                )
                se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, SEARCH_ENGINE_TOP_K)

        elif dialogue_mode == "安全问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, MERGED_MAX_DOCS_NUM)
                score_threshold = st.number_input(
                    "知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01
                )

        elif dialogue_mode == "融合问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, MERGED_MAX_DOCS_NUM)
                score_threshold = st.number_input(
                    "知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01
                )

        elif dialogue_mode == "融合问答v2":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, MERGED_MAX_DOCS_NUM)
                score_threshold = st.number_input(
                    "知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01
                )
                
        elif dialogue_mode == "流式问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                new_kb_list = sorted(kb_list,key=to_pinyin_key)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    new_kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, MERGED_MAX_DOCS_NUM)
                score_threshold = st.number_input(
                    "知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01
                )

    # Display chat messages from history on app rerun

    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("正在思考...")
            text = ""
            r = api.chat_chat(
                prompt, history=history, model=llm_model, temperature=temperature
            )
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标
        elif dialogue_mode == "自定义Agent问答":
            chat_box.ai_say([
                f"正在思考和寻找工具 ...",])
            text = ""
            element_index = 0
            for d in api.agent_chat(prompt,
                                    history=history,
                                    model=llm_model,
                                    temperature=temperature):
                try:
                    d = json.loads(d)
                except:
                    pass
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)

                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
                elif chunk := d.get("tools"):
                    element_index += 1
                    chat_box.insert_msg(Markdown("...", in_expander=True, title="使用工具...", state="complete"))
                    chat_box.update_msg("\n\n".join(d.get("tools", [])), element_index=element_index, streaming=False)
            chat_box.update_msg(text, element_index=0, streaming=False)
        elif dialogue_mode == "知识库问答":
            chat_box.ai_say([
                f"正在查询知识库 `{selected_kb}` ...",
                Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
            ])
            text = ""
            for d in api.knowledge_base_chat(
                prompt,
                selected_kb,
                kb_top_k,
                score_threshold,
                history,
                model=llm_model,
                temperature=temperature,
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
        elif dialogue_mode == "搜索引擎问答":
            chat_box.ai_say([
                f"正在执行 `{search_engine}` 搜索...",
                Markdown("...", in_expander=True, title="网络搜索结果", state="complete"),
            ])
            text = ""
            for d in api.search_engine_chat(prompt,
                                            search_engine_name=search_engine,
                                            top_k=se_top_k,
                                            history=history,
                                            model=llm_model,
                                            temperature=temperature):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
        elif dialogue_mode == "流式问答":
            chat_box.ai_say([
                f"正在查询知识库 `{selected_kb}` ...",
                Markdown("...", in_expander=True, title="知识库匹配结果", state="complete"),
            ])
            text = ""
            for d in api.career_flow_chat(prompt,
                                          selected_kb,
                                          kb_top_k,
                                            score_threshold,
                                            history=history
                                            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
            "清空对话",
            use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
