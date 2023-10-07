from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
from fastapi import Body, Request
import jieba

model_path = "/home/ubuntu/bertTrain/testoutput/checkpoint-600"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 想要用页面看的功能
"选科要求"

#



def recommend_suggest(
    query: str = Body(..., examples=["samples"]), test_key: str = Body("faiss")
):
    """
    对获取的query进行判断
    1、是否是当前意图
    2、有没有命中一些推荐的内容

    Returns:
        _type_: _description_
    """

    cut_res = jieba.lcut(query, cut_all=False)

    ret = {"recommend_target": []}

    res = classifier(query)
    final_res = label_dict[res[0]["label"]]
    print(final_res)
    ret["answer"] = final_res
    return JSONResponse(ret)
