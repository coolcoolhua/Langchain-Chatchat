import streamlit as st
from configs.server_config import FSCHAT_MODEL_WORKERS
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs.model_config import (
    llm_model_dict,
    LLM_MODEL,
    MERGED_MAX_DOCS_NUM,
    PROMPT_TEMPLATE,
)
from server.utils import get_model_worker_config
from typing import List, Dict


chat_box = ChatBox(
    assistant_avatar=os.path.join("img", "chatchat_icon_blue_square_v2.png")
)

judge_template1 = """你是一个生涯教育老师。判断以下问题是「闲聊问题」还是和生涯教育相关的「观点问题」还是「事实问题」。
示例问题：“你好”，示例答案：“「闲聊问题」”。
示例问题：“我适合学心理学吗”，示例答案：“「观点问题」”。
示例问题：“什么是心理学”，示例答案：“「事实问题」”。
问题：“{}”, 答案:
"""

truth_template = """<角色>你是一个生涯教育老师</角色>
<指令>优先从已知信息提取答案。
如果无法从中得到答案，忽略已知内容直接回答问题。
回答尽量详细，不要出现“角色”，“指令”，“已知信息”内的内容。</指令>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>                 
"""

opinion_template = """<角色>你是一个生涯教育老师</角色>
<指令>优先从已知信息提取答案。
以客观中立的态度，在回答中以“正面”和”反面“两个方面进行回答，最后附上总结。
如果无法从中得到答案，忽略已知内容直接回答问题。
回答尽量详细，不要出现“角色”，“指令”，“已知信息”内的内容。</指令>
<已知信息>{{ context }}</已知信息>
<问题>{{ question }}</问题>
"""

chat_template = """<角色>你是一个生涯教育老师。你擅长教导学生进行生涯探索。你的语言风格亲切。</角色>
<限制>不允许说自己是一个人工智能模型。</限制>
<问题>{}</问题>
"""

unsatisfy_prompt = ""


def get_messages_history(history_len: int) -> List[Dict]:
    def filter(msg):
        """
        针对当前简单文本对话，只返回每条消息的第一个element的内容
        """
        content = [
            x._content
            for x in msg["elements"]
            if x._output_method in ["markdown", "text"]
        ]
        return {
            "role": msg["role"],
            "content": content[0] if content else "",
        }

    history = chat_box.filter_history(
        100000, filter
    )  # workaround before upgrading streamlit-chatbox.
    user_count = 0
    i = 1
    for i in range(1, len(history) + 1):
        if history[-i]["role"] == "user":
            user_count += 1
            if user_count >= history_len:
                break
    return history[-i:]


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
            ["融合问答v2", "搜索引擎问答","LLM 对话"],
            on_change=on_mode_change,
            key="dialogue_mode",
        )

        def on_llm_change():
            st.session_state["prev_llm_model"] = llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x
        
        def reanswer1():
            text = ""
            for dt in api.search_engine_chat(unsatisfy_prompt, "bing", 3, model=llm_model):
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

        used_template = PROMPT_TEMPLATE
        # used_template = st.text_area("判断问题类型使用的prompt是", PROMPT_TEMPLATE, height=200)
        judge_template_save = st.text_area(
            "判断问题类型 使用的prompt是", judge_template1, height=200
        )
        if st.button("这个prompt很棒，保存一下", key="used", use_container_width=True):
            save_prompt(judge_template_save, "判断")
        chat_template_save = st.text_area("闲聊问题 使用的prompt是", chat_template, height=200)
        if st.button("这个prompt很棒，保存一下", key="judge", use_container_width=True):
            save_prompt(judge_template_save, "闲聊")
        truth_template_save = st.text_area(
            "事实问题 使用的prompt是", truth_template, height=200
        )
        if st.button("这个prompt很棒，保存一下", key="truth", use_container_width=True):
            save_prompt(truth_template_save, "事实")
        opinion_template_save = st.text_area(
            "观点问题 使用的prompt是", opinion_template, height=200
        )
        if st.button("这个prompt很棒，保存一下", key="opinion", use_container_width=True):
            save_prompt(opinion_template_save, "观点")

        running_models = api.list_running_models()
        config_models = api.list_config_models()
        for x in running_models:
            if x in config_models:
                config_models.remove(x)
        llm_models = running_models + config_models
        if "prev_llm_model" not in st.session_state:
            index = llm_models.index(LLM_MODEL)
        else:
            index = 0
        llm_model = st.selectbox(
            "选择LLM模型：",
            llm_models,
            index,
            format_func=llm_model_format_func,
            on_change=on_llm_change,
            # key="llm_model",
        )
        if st.session_state.get(
            "prev_llm_model"
        ) != llm_model and not get_model_worker_config(llm_model).get("online_api"):
            with st.spinner(f"正在加载模型： {llm_model}"):
                r = api.change_llm_model(
                    st.session_state.get("prev_llm_model"), llm_model
                )
            st.session_state["prev_llm_model"] = llm_model

        history_len = st.number_input("历史对话轮数：", 0, 10, HISTORY_LEN)
        
        

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
                score_threshold = st.number_input(
                    "知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01
                )
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

    # Display chat messages from history on app rerun

    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter "

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("正在思考...")
            text = ""
            r = api.chat_chat(prompt, history=history, model=llm_model)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标
        elif dialogue_mode == "问答判断":
            chat_box.ai_say("正在思考...")
            text = ""
            r = api.chat_judge(prompt, history=[], model=llm_model)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标

        elif dialogue_mode == "知识库问答":
            history = get_messages_history(history_len)
            chat_box.ai_say(
                [
                    f"正在查询知识库 `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="知识库匹配结果"),
                ]
            )
            text = ""
            for d in api.knowledge_base_chat(
                prompt, selected_kb, kb_top_k, score_threshold, history, model=llm_model
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                text += d["answer"]
                chat_box.update_msg(text, 0)
                chat_box.update_msg("\n\n".join(d["docs"]), 1, streaming=False)
            chat_box.update_msg(text, 0, streaming=False)
        elif dialogue_mode == "搜索引擎问答":
            chat_box.ai_say(
                [
                    f"正在执行 `{search_engine}` 搜索...",
                    Markdown("...", in_expander=True, title="网络搜索结果"),
                ]
            )
            text = ""
            for d in api.search_engine_chat(
                prompt, search_engine, se_top_k, model=llm_model
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                else:
                    text += d["answer"]
                    chat_box.update_msg(text, 0)
                    chat_box.update_msg("\n\n".join(d["docs"]), 1, streaming=False)
            chat_box.update_msg(text, 0, streaming=False)
        elif dialogue_mode == "安全问答":
            history = get_messages_history(history_len)
            # history = []
            chat_box.ai_say(
                [
                    f"正在查询知识库并过滤安全词 `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="知识库匹配结果"),
                ]
            )
            text = ""
            docs = ""
            for d in api.kb_safe_chat(
                prompt, selected_kb, kb_top_k, score_threshold, history
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                docs += d["docs"]
                text += d["answer"]
                # chat_box.update_msg(text, 0)
            chat_box.update_msg(docs, 1, streaming=False)
            chat_box.update_msg(text, 0, streaming=False)
        elif dialogue_mode == "融合问答":
            history = get_messages_history(history_len)
            chat_box.ai_say(
                [
                    f"正在查询知识库并过滤安全词 `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="知识库匹配结果"),
                ]
            )
            text = ""
            docs = ""
            for d in api.merged_chat_diytemplate(
                prompt, selected_kb, used_template, kb_top_k, score_threshold, history
            ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                docs = d["docs"]
                text += d["answer"]
                # chat_box.update_msg(text, 0)

            chat_box.update_msg("".join(docs), 1, streaming=False)
            chat_box.update_msg(text, 0, streaming=False)

        elif dialogue_mode == "融合问答v2":
            history = get_messages_history(history_len)
            text = ""
            docs = ""
            # step1 判断是哪种类型的问题
            judge_template = judge_template_save.format(prompt)
            r = api.chat_judge(judge_template, history=history, model=llm_model)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t
            st.toast("问题类型判断结果为:{}".format(text))
            if "闲聊" in text:
                chat_box.ai_say("正在思考...")
                text = ""
                chat_prompt = chat_template_save.format(prompt)
                for d in api.chat_judge(chat_prompt, history=history, model=llm_model):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    text += d
                chat_box.update_msg(text, streaming=False)
            else:
                unsatisfy_prompt = prompt
                if "事实" in text:
                    used_template = truth_template_save
                elif "观点" in text:
                    used_template = opinion_template_save
                text = ""
                chat_box.ai_say(
                    [
                        f"正在回答 `{selected_kb}` ...",
                        Markdown("...", in_expander=True, title="知识库匹配结果"),
                    ]
                )
                for d in api.merged_chat_diytemplate(
                    prompt,
                    selected_kb,
                    used_template,
                    kb_top_k,
                    score_threshold,
                    history,
                ):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    docs = d["docs"]
                    text += d["answer"]
                chat_box.update_msg("".join(docs), 1, streaming=False)
                chat_box.update_msg(text, 0, streaming=False)
                
                
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
