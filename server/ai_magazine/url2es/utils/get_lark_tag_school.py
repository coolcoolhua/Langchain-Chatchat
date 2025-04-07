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
df = pd.read_csv('./server/ai_magazine/url2es/utils/飞书标签映射表_院校.csv')
# df只保留'展示名称','id'两列
df = df[['展示名称','ID']]
# df转为dict, "展示名称"为key, "ID"为value
df = df.set_index('展示名称').T.to_dict('list')
# 去掉外层的list
for key in df.keys():
    df[key] = df[key][0]
    
# 读取csv
df1 = pd.read_csv('./server/ai_magazine/url2es/utils/飞书标签映射表_院校.csv')
# df只保留'展示名称','id'两列
df1 = df1[['展示名称','ID']]
# df转为dict, "展示名称"为key, "ID"为value
df1 = df1.set_index('ID').T.to_dict('list')
# 去掉外层的list
for key in df1.keys():
    df1[key] = df1[key][0]
    
def get_lark_tag_school(target_website):
    
    """
    
    返回学校名对应的学校ID
    
    Returns:
        _type_: _description_
    """
    
    return {"_id": str(df[target_website])}

def get_lark_tag_school_name(idnumber):
    idnumber = int(idnumber)
    return str(df1[idnumber])





if __name__ == '__main__':
    res = get_lark_tag_school_name(1785687387030665)
    print(res)