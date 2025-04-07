from fastapi import Body
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from typing import *
import re
import time
import requests


ALL_RETURNED_MAJOR_NUM = 3
RETURNED_MAJOR_NUM = 1
RETURNED_SCHOOL_HISTORY_NUM = 0


def post_lark(records):
    url = "https://ae-openapi.feishu.cn/auth/v1/appToken"
    payload = {"clientId":"c_f917192b155c47449b6a","clientSecret": "cd4ce275ab1e4e349e942e6d96d1563c"}
    headers = {
    "Content-Type": "application/json"
    }
    response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
    # 写入接口
    token = response.json()["data"]['accessToken']
    
    # 创建推荐记录
    
    
    
    url = "https://ae-openapi.feishu.cn/v1/data/namespaces/careerContentManagement__c/objects/recommendationRecord/records"
    payload = {
        "record":
            {
                "relatedMajor": [
                    {"_id":"100"}
                    ],
                "relatedUniversity":[
                    {"_id":"100"}
                    ],
                    "text_64c5e7f2614":"Sample text",
                    "userID":"Sample text",
                    "userName":"Sample text"
            }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": token
    }
    response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
    print(response.text)


def get_article_res(article):
    if 'yixi.tv' in article["_source"]["originalUrl"]:
        hid = article["_source"]["originalUrl"].split("https://yixi.tv/api/site/draft/?type=0&")[-1].replace("id=","")
        article["_source"]["originalUrl"] = "https://www.yixi.tv/wx/h5/#/videos/?video_type=0&video_id={}&album_id=0".format(hid)
    res = {
        "record_id": article["_source"]["id"],
        "contentUUID": article["_source"]["contentUUID"],
        "title": article["_source"]["title"],
        "article": article["_source"]["contentUrl"],
        "relation": {
            "link": article["_source"]["originalUrl"],
            "title": article["_source"]["title"],
            "introduction": article["_source"]["description"]
        },
        "article_type" : article["_source"]["topicTags"]["name"],
        "major_name": [t["name"] for t in article["_source"]["relatedMajors"]],
        "major_id": [t["id"] for t in article["_source"]["relatedMajors"]],
        "university_name": [t["name"] for t in article["_source"]["relatedUniversities"]],
        "university_id": [t["id"] for t in article["_source"]["relatedUniversities"]],
        "contentSize":  int(article["_source"].get("contentSize", 0))
    }
    return res


def get_article_quality_check(major, articles):
    # 如果文章数量不够, 记录日志
    if len(articles) < RETURNED_MAJOR_NUM:
        with open("server/ai_magazine/推荐记录.txt", "a") as f:
            check_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(check_time + " " + major + "文章数量不足\n")
            

# 尽量先保证有图
def get_search_query(major_name, type_name, usedUUIDs):
    query = {
            "size": RETURNED_MAJOR_NUM,  # 指定返回的文档数量
            "query": {
                "bool": {
                    "must": [
                        {"match": {"relatedMajors.name.keyword": major_name}},
                        {"match": {"topicTags.name.keyword": type_name}},
                        {"match": {"reviewStatus.keyword": "reviewed"}},
                        {"range": {
                            "contentSize": {"lt": 5000000}  # contentSize是整数类型,直接比较数值
                        }},
                        # {"exists": {"field": "contentSize"}}
                    ],
                    "must_not": [
                        {
                            "terms": {"contentUUID.keyword": usedUUIDs}
                        },
                        {
                            "term": {"reviewStatus.keyword": "toBeRevised"}
                        },
                        {
                            "term": {"reviewStatus.keyword": "abandon"}
                        },
                        {
                            "term": {"reviewStatus.keyword": "toBeReviewed"}
                        }
                    ],
                }
            },
            "sort": [
                {"contentSize": {"order": "asc"}},
            ]
        }
    return query

# 兜底策略，如果找不到的话就补充
def get_fallback_strategy(major_name, type_name, usedUUIDs):
    print("兜底策略")
    query = {
        "size": RETURNED_MAJOR_NUM,  # 指定返回的文档数量
        "query": {
            "bool": {
                "must": [
                    {"match": {"relatedMajors.name.keyword": major_name}},
                    {"match": {"topicTags.name.keyword": type_name}},
                    {"match": {"reviewStatus.keyword": "reviewed"}},
                    {"range": {
                        "contentSize": {"lt": 5000000}  # contentSize是整数类型,直接比较数值
                    }},
                    # {"exists": {"field": "contentSize"}}
                ],
                "must_not": [
                    {
                        "terms": {"contentUUID.keyword": usedUUIDs}
                    },
                    {
                        "term": {"reviewStatus.keyword": "toBeRevised"}
                    },
                    {
                        "term": {"reviewStatus.keyword": "abandon"}
                    },
                    {
                        "term": {"reviewStatus.keyword": "toBeReviewed"}
                    }
                ],
            }
        },
        "sort": [
            {"contentSize": {"order": "desc"}},
        ]
    }
    return query


# 连接ES
es = Elasticsearch(
    ["http://localhost:9200"],
    sniff_on_start=False,  # 连接前测试
    sniff_on_connection_fail=True,  # 节点无响应时刷新节点
    sniffer_timeout=60,  # 设置超时时间
)

def get_recommend_strategy(major_nums,every_major_article_num,remainder):
    
    # 每个专业推1篇
    strategy = [1 for t in range(major_nums)]
    return strategy
    

def articles_reassign():
    pass


def get_recommend_articles(
    collegeTags: Dict = Body(
        {},
        description="院校相关标签",
        examples=[
            {
                "意向学校": ["清华大学"],
                "推荐学校": {"冲刺学校": ["北京大学"], "稳妥学校": ["中国人民大学"], "保底学校": ["北京师范大学"]},
            }
        ],
    ),
    majorTags: Dict = Body(
        {},
        description="院校相关标签",
        examples=[
            {
                "意向专业": ["计算机科学与技术"],
            }
        ],
    ),
    usedUUIDs: List = Body(
        [],
        description="已经使用过的文章UUID",
        examples=[
            ["0b6df52aa3d811eea4737af451f953bb", "0a6ebca4a3d811eea4737af451f953bb"]
        ],
    ),
):
    """


    Args:
        collegeTags (_type_, optional): _description_. Defaults to Body({}, description="院校相关标签", examples=[ { "意向学校":["清华大学"], "推荐学校": { "冲刺学校":["北京大学"], "稳妥学校":["中国人民大学"], "保底学校":["北京师范大学"] } }] ).
        majorTags (_type_, optional): _description_. Defaults to Body({}, description="院校相关标签", examples=[ { "意向专业":["计算机科学与技术"], }] ).
        usedUUIDs (List, optional): _description_. Defaults to Body([], description="已经使用过的文章UUID", examples=[["1234567890","12345678901"]]).

    Returns:
        _type_: _description_
    """
    
    start = time.time()
    print(collegeTags, majorTags, usedUUIDs)
    
    # 每个专业分配的文章数量，用ALL_RETURNED_MAJOR_NUM / len(majorTags["意向专业"])计算
    # 余数分配给前几个专业
    every_major_article_num = ALL_RETURNED_MAJOR_NUM // len(majorTags["意向专业"])
    # 余数
    remainder = ALL_RETURNED_MAJOR_NUM % len(majorTags["意向专业"])
    # 每个专业分配的数量
    if len (majorTags["意向专业"]) > ALL_RETURNED_MAJOR_NUM:
        print("意向专业数量大于12")
        majorTags["意向专业"] = majorTags["意向专业"][:12]
    if len(majorTags["意向专业"]) < 2:
        articles_num_list = [6 for t in majorTags["意向专业"]]
    else:
        articles_num_list = get_recommend_strategy(len(majorTags["意向专业"]),every_major_article_num,remainder)
        
    print("每个专业分配的文章数量", articles_num_list)

    # 共有3个模块 
    # 1.专业探索, 查找所有relatedMajors和意向专业相同, 且topicTags为专业重点扫盲的文章
    all_major_explorer_articles = []
    major_explorer_articles = {}
    # 构建查询
    for major in majorTags["意向专业"]:
        major_explorer_articles = {"title": major + "重点扫盲", "major":major, "children": []}
        print("意向专业", major)
        query = get_search_query(major, "专业重点扫盲", usedUUIDs)
        res = es.search(index="shijie", body=query)
        print("要排除的ids", usedUUIDs)
        # 如果没有找到文章,使用兜底策略再搜一遍
        # if len(res["hits"]["hits"]) == 0:
        #     query = get_fallback_strategy(major, "专业重点扫盲", usedUUIDs)
        #     res = es.search(index="shijie", body=query)
        if len(res["hits"]["hits"]) > 0:
            for article in res["hits"]["hits"]:
                major_explorer_articles["children"].append(get_article_res(article))
                usedUUIDs.append(article["_source"]["contentUUID"])
            all_major_explorer_articles.append(major_explorer_articles)

    # 2.专业杰出人物, 查找所有relatedMajors和意向专业相同, 且topicTags为专业杰出人物的文章
    major_outstanding_articles = {}
    all_major_outstanding_articles = []
    # 构建查询
    for major in majorTags["意向专业"]:
        major_outstanding_articles = {"title": major + "人物故事", "major":major, "children": []}
        query = get_search_query(major, "专业杰出人物", usedUUIDs)
        res = es.search(index="shijie", body=query)
        # if len(res["hits"]["hits"]) == 0:
        #     query = get_fallback_strategy(major, "专业杰出人物", usedUUIDs)
        #     res = es.search(index="shijie", body=query)
        if len(res["hits"]["hits"]) > 0:
            for article in res["hits"]["hits"]:
                major_outstanding_articles["children"].append(get_article_res(article))
                usedUUIDs.append(article["_source"]["contentUUID"])
            all_major_outstanding_articles.append(major_outstanding_articles)
    # 3.专业前沿资讯, 查找所有relatedMajors和意向专业相同, 且topicTags为专业前沿资讯的文章
    major_frontier_articles = {}
    all_major_frontier_articles = []
    # 构建查询
    for major in majorTags["意向专业"]:
        major_frontier_articles = {"title": major + "前沿资讯", "major":major, "children": []}
        query = get_search_query(major, "专业前沿资讯", usedUUIDs)
        res = es.search(index="shijie", body=query)
        # if len(res["hits"]["hits"]) == 0:
        #     query = get_fallback_strategy(major, "专业前沿资讯", usedUUIDs)
        #     res = es.search(index="shijie", body=query)
        if len(res["hits"]["hits"]) > 0:
            for article in res["hits"]["hits"]:
                major_frontier_articles["children"].append(get_article_res(article))
                usedUUIDs.append(article["_source"]["contentUUID"])
            all_major_frontier_articles.append(major_frontier_articles)
        

    # 返回模版
    ret = {"interest_explore": []}
    for t in majorTags["意向专业"]:
        if t not in ret["interest_explore"]:
            temp = {"title": t+"相关文章", "children": []}
            ret["interest_explore"].append(temp)
            
            
    all_major_articles = {}
    # 将major整理为dict
    for t in all_major_explorer_articles:
        if t["major"] not in all_major_articles:
            all_major_articles[t["major"]] = []
            all_major_articles[t["major"]].extend(t["children"])
        else:
            all_major_articles[t["major"]].extend(t["children"])
    for t in all_major_outstanding_articles:
        if t["major"] not in all_major_articles:
            all_major_articles[t["major"]] = []
            all_major_articles[t["major"]].extend(t["children"])
        else:
            all_major_articles[t["major"]].extend(t["children"])
    for t in all_major_frontier_articles:
        if t["major"] not in all_major_articles:
            all_major_articles[t["major"]] = []
            all_major_articles[t["major"]].extend(t["children"])
        else:
            all_major_articles[t["major"]].extend(t["children"])
    
    for t in all_major_articles:
        print(t, len(all_major_articles[t]))        
    
    print("all_major_articles", all_major_articles.keys())
    print("articles_num_list", articles_num_list)
    # 按照分配策略，每个专业分配对应数量的文章
    num_check = 0 
    for i in range(len(majorTags["意向专业"])):
        major_name = majorTags["意向专业"][i]
        if major_name in all_major_articles and all_major_articles[major_name]:
            #如果够2篇以上
            if num_check > 2:
                num_check = 0 
                break
            ret["interest_explore"][i]["children"].extend(all_major_articles[major_name][:articles_num_list[i]])
            num_check +=1
            
    # 只保留children不为空的
    final_ret = {
        "interest_explore": []
    }
    for t in ret["interest_explore"]:
        if len(t["children"]) > 0:
            final_ret["interest_explore"].append(t)
    print("here",len(final_ret["interest_explore"]))
    # 计算所有文章内容大小总和
    total_content_size = 0
    for article_group in final_ret["interest_explore"]:
        print(article_group["title"], len(article_group["children"]))
        for article in article_group["children"]:
            if "contentSize" in article:
                print('标题',article['title'])
                print('大小',article["contentSize"])
                total_content_size += article["contentSize"]
    
    final_ret["total_content_size"] = total_content_size
    
    end = time.time()
    print("文章推荐用时",end-start)
    
    # ret["interest_explore"].extend(all_major_explorer_articles)
    # ret["interest_explore"].extend(all_major_outstanding_articles)
    # ret["interest_explore"].extend(all_major_frontier_articles)
    # ret["interest_explore"].extend(all_university_history_articles)
    # ret["interest_explore"].extend(all_university_policy_articles)

    return JSONResponse(final_ret)