# from transformers import pipeline
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
from fastapi import Body, Request

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_path = '/home/ubuntu/.cache/modelscope/hub/damo/nlp_structbert_sentiment-classification_chinese-base'
semantic_cls = pipeline(Tasks.text_classification, 'damo/nlp_structbert_sentiment-classification_chinese-base')

def bert_sentiment_analysis(query: str = Body(..., examples=["samples"]),test_key: str = Body("faiss")):
  print("query=",query)
  ret = {
    "answer": ""
  }
  res = semantic_cls(query)
  final_res = res['labels'][0]
  print(final_res)
  ret["answer"]=final_res
  return JSONResponse(ret)
