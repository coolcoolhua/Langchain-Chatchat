import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from server.chat import (chat, knowledge_base_chat, openai_chat,
                         search_engine_chat, kb_safe_chat, 
                         context_chat,
                         chat_judge,
                         merged_chat,
                         search_engine_docs,
                         unsatisfy_question_chat,
                         bert_chat_judge,
                         bert_truth_judge,
                         bert_relevance_judge,
                         bert_sentiment_analysis,
                         search_engine_chat, agent_chat,
                         chat_stop,
                         career_flow_chat,
                         career_flow_chat_merged,
                         career_flow_chat_sl,
                         get_content_tags,
                         get_content_tags_stream,
                         es_add_data,
                         get_recommend_articles,
                         context_chat_tag
                         )
from server.rank_prediction import score_insight_train, score_insight_prediction

from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
from server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                              update_docs, download_doc, recreate_vector_store,
                                              search_docs, DocumentWithScore)
from server.llm_api import list_running_models, list_config_models, change_llm_model, stop_llm_model
from server.utils import BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline
from typing import List

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    return RedirectResponse(url="/docs")


def create_app():
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: Chat
    app.post("/chat/fastchat",
             tags=["Chat"],
             summary="与llm模型对话(直接与fastchat api对话)")(openai_chat)

    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)")(chat)
    app.post("/chat/chat_stop",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)")(chat_stop)
    app.post("/chat/chat_judge",
            tags=["Chat"],
            summary="判断是哪种问题)")(chat_judge)
    
    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="与知识库对话")(knowledge_base_chat)

    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="与搜索引擎对话")(search_engine_chat)
    app.post("/chat/career_flow_chat",
             tags=["Chat"],
             summary="生涯流式问答,包括意图处理，敏感词处理")(career_flow_chat)
    app.post("/chat/career_flow_chat_merged",
             tags=["Chat"],
             summary="生涯流式问答(不划分专业),包括意图处理，敏感词处理")(career_flow_chat_merged)
    app.post("/chat/career_flow_chat_sl",
             tags=["Chat"],
             summary="生涯流式问答(sl版本)")(career_flow_chat_sl)
    
    app.post("/chat/search_engine_docs",
             tags=["Chat"],
             summary="返回搜索引擎文档和上下文")(search_engine_docs)
    
    app.post("/chat/bert_chat_judge",
             tags=["Chat"],
             summary="bert判断是闲聊意图还是职业意图")(bert_chat_judge)
    app.post("/chat/bert_truth_judge",
             tags=["Chat"],
             summary="bert判断是事实意图还是观点意图")(bert_truth_judge)
    app.post("/chat/bert_relevance_judge",
             tags=["Chat"],
             summary="bert判断是事实意图还是观点意图")(bert_relevance_judge)
    
    app.post("/chat/bert_sentiment_analysis",
             tags=["Chat"],
             summary="bert判断是事实意图还是观点意图")(bert_sentiment_analysis)
    app.post("/chat/get_content_tags",
             tags=["Chat"],
             summary="大模型给内容打标签")(get_content_tags)
    app.post("/chat/get_content_tags_stream",
             tags=["Chat"],
             summary="大模型给内容打标签流")(get_content_tags_stream)
    app.post("/chat/context_chat_tag",
             tags=["Chat"],
             summary="搜索增加后让大模型给内容打标签流")(context_chat_tag)
    
    # es相关操作
    app.post("/chat/es_add_data",
            tags=["ES"],
            summary="es数据库新增数据")(es_add_data)
    app.post("/chat/get_recommend_articles",
            tags=["ES"],
            summary="推荐文章")(get_recommend_articles)
    
    
    app.post("/chat/kb_safe_chat",
             tags=["Chat"],
             summary="知识库问答+敏感词过滤")(kb_safe_chat)
    app.post("/chat/merged_chat",
            tags=["Chat"],
            summary="知识库问答+敏感词过滤")(merged_chat)
    app.post("/chat/context_chat",
             tags=["Chat"],
             summary="知识库问答+敏感词过滤+自定义prompt")(context_chat)
    app.post("/chat/answer_again",
             tags=["Chat"],
             summary="对于不满意的回答重新用搜索引擎回答")(unsatisfy_question_chat)

    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="与agent对话")(agent_chat)
    
    
    # Tag: 学业透视
    app.post("/rank/score_insight_prediction",
             tags=["Rank"],
             summary="学业透视预测模块")(score_insight_prediction)
    app.post("/rank/score_insight_train",
             tags=["Rank"],
             summary="学业透视新沂版挎包模块")(score_insight_train)

    # Tag: Knowledge Base Management
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库列表")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="创建知识库"
             )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库"
             )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库内的文件列表"
            )(list_files)

    app.post("/knowledge_base/search_docs",
             tags=["Knowledge Base Management"],
             response_model=List[DocumentWithScore],
             summary="搜索知识库"
             )(search_docs)

    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="上传文件到知识库，并/或进行向量化"
             )(upload_docs)

    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库内指定文件"
             )(delete_docs)

    app.post("/knowledge_base/update_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新现有文件到知识库"
             )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="下载对应的知识文件")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
             tags=["Knowledge Base Management"],
             summary="根据content中文档重建向量库，流式输出处理进度。"
             )(recreate_vector_store)

    # LLM模型相关接口
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="列出当前已加载的模型",
             )(list_running_models)

    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="列出configs已配置的模型",
             )(list_config_models)

    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="停止指定的LLM模型（Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="切换指定的LLM模型（Model Worker)",
             )(change_llm_model)

    return app


app = create_app()


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        # uvicorn.run('api:app', host=host, port=port, workers=4 )
        uvicorn.run('api:app', host=host, port=port, reload= True )
        # uvicorn.run(app,host=host,port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain'
                                                 ' ｜ 基于本地知识库的 ChatGLM 问答')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)
    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
