from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
from fastapi import Body, Request


def score_insight_prediction(
    model_id: str = Body(..., examples=["123456"]),
    data: list = Body([], examples=[
      [
        {
          "学生id": 123456,
          "学生本次成绩": 123
        },
        {
          "学生id": 654321,
          "学生本次成绩": 320
        }
    ]])
):
  
  
    ret = [
      {
        "uuid": 123456,
        "进步名次": 30,
        "本次成绩预测最低排名": 120,
        "本次成绩预测最高排名": 90,
        "进步30名后的最低排名": 1,
        "进步30名后的最高": 1,
      },
      {
        "uuid": 654321,
        "进步名次": 300,
        "本次成绩预测最低排名": 120,
        "本次成绩预测最高排名": 90,
        "进步30名后的最低排名": 1,
        "进步30名后的最高": 1,
      },
      
    ]
    return JSONResponse(ret)
