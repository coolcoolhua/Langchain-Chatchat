import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import pandas as pd
import openpyxl
import os


# 读取csv
df = pd.read_csv('./server/ai_magazine/url2es/utils/飞书标签映射表_专业.csv')
# 保留所有级别为"专业大类"和"学门"的数据
df = df[df['级别'].str.contains("专业大类") | df['级别'].str.contains("学门")]
# df只保留'展示名称','id'两列
df = df[['展示名称','ID']]
# 专业大类的"学门"两字去掉
df['展示名称'] = df['展示名称'].str.replace("学门:","")
# df转为dict, "展示名称"为key, "ID"为value
df = df.set_index('展示名称').T.to_dict('list')
# 去掉外层的list
for key in df.keys():
    df[key] = df[key][0]

def get_lark_tag_major(target_major):
    
    """
    
    返回学校名对应的学校ID
    
    Returns:
        _type_: _description_
    """
    
    if df.get(target_major) is None:
        return {}
    return {"_id": str(df[target_major])}


if __name__ == '__main__':
    res = get_lark_tag_major('植物保护类')
    print(res)