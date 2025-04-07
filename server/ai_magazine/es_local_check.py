from elasticsearch import Elasticsearch

es = Elasticsearch(
        ["http://localhost:9200"],
        sniff_on_start=False,            # 连接前测试
        sniff_on_connection_fail=True,  # 节点无响应时刷新节点
        sniffer_timeout=60              # 设置超时时间
    )


# 找到索引shijie下的所有内容
def es_search_all(index):
    res = es.search(index=index, body={"query": {"match_all": {}}})
    print(res)
    return res

# 删除某个索引
def es_delete_index(index):
    res = es.indices.delete(index=index)
    print(res)
    return res

# 创建某个索引
def es_create_index(index):
    res = es.indices.create(index=index)
    print(res)
    return res


# 查询是否存在某条数据的id字段
def es_search_id(index, id):
    res = es.search(index=index, body={"query": {"match": {"id": id}}})
    print(res)
    return res


# 进行某个查询
def es_search_test():
    # 查询index为shijie下,relatedMajors.name.keyword字段为计算机科学与技术,且topicTags.name.keyword字段为专业重点扫盲的文章
    query = {
        "size": 10,  # 指定返回的文档数量
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "relatedMajors.name.keyword": "计算机科学与技术"
                        }
                    },
                    {
                        "match": {
                            "topicTags.name.keyword": "专业重点扫盲"
                        }
                    }
                ]
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    print(len(res["hits"]["hits"]))


# es_search_all("shijie")
# es_delete_index("shijie")
# es_create_index("school")
# es_create_index("major")
# es_create_index("province")
# es_search_id("shijie", "234")
# es_search_test()

# import pandas as pd
# major_df = pd.read_csv('专业名称.csv')
# for index, row in major_df.iterrows():
#     if index % 10 ==0:
#         print(index)
#     es.index(index='major', doc_type='doc', body=row.to_dict())


# major = '医学'
# res = es.search(index='major', body={
#     "query": {
#         "match": {
#             "专业名称": {
#                 "query": major,
#                 "operator": "and"  # 要求匹配的词语都存在
#             }
#         }
#     },
#     "size":100
# })
# for hit in res['hits']['hits']:
#     print(hit['_source']['专业名称'])





# res = es.search(index='school', body
#                 ={"query": {"match": {"院校名称": "北京大学"}}})
# for hit in res['hits']['hits']:
#     print(hit['_source']['院校名称'])
    


# es_delete_index('school')
# es_create_index('school')

# import pandas as pd
# school_df = pd.read_csv('院校名称.csv')
# for index, row in school_df.iterrows():
#     if index % 10 ==0:
#         print(index)
#     es.index(index='school', doc_type='doc', body=row.to_dict())


# es_create_index('major')
# es_delete_index('province')
# es_create_index('province')


# import pandas as pd
# province_df = pd.read_csv('省份名称.csv')
# # 保留name和province_name
# # province_df = province_df[['name', 'province_name']]

# # 索引添加数据
# for index, row in province_df.iterrows():
#     if index % 10 ==0:
#         print(index)
#     # 名称，省份
#     es.index(index='province', doc_type='doc', body=row.to_dict())



# es_delete_index('major')
# es_create_index('major')

# import pandas as pd
# major_df = pd.read_csv('专业名称.csv')
# for index, row in major_df.iterrows():
#     if index % 10 ==0:
#         print(index)
#     es.index(index='major', doc_type='doc', body=row.to_dict())

# 查找shijie这个index下有多少条数据
# def es_search_all(index):
#     res = es.search(index=index, body={"query": {"match_all": {}}})
#     print(res)
#     return res

# print(len(es_search_all("shijie")["hits"]["hits"]))

# print(es.count(index='shijie')['count'])


# res = es.search(index='shijie', body={
#     "query": {
#     "term": {
#       "contentUUID": "a69fbeeebce611eea9057af451f953bb"
#     }
#   }
# })



# major = '教育学'
# res = es.search(index='major', body={
#     "query": {
#         "match": {
#             "专业名称": {
#                 "query": major,
#                 "operator": "and"  # 要求匹配的词语都存在
#             }
#         }
#     },
#     "size":100
# })
# for hit in res['hits']['hits']:
#     print(hit['_source']['专业名称'])

# print(res)


# 获取index的mapping信息
def get_index_mapping(index):
    mapping = es.indices.get_mapping(index=index)
    print(f"\n{index}索引的字段映射:")
    print(mapping)
    return mapping

# 获取一条示例文档
def get_sample_doc(index):
    res = es.search(index=index, body={"query": {"match_all": {}}, "size": 1})
    if res['hits']['hits']:
        print(f"\n{index}索引的示例文档:")
        print(res['hits']['hits'][0]['_source'])
    return res
# 获取一席和未来科学大奖官网的sourceTags统计信息
def get_source_tags_stats():
    query = {
        "size": 0,
        "query": {
            "bool": {
                "should": [
                    {"match": {"sourceTags.name.keyword": "一席"}},
                    {"match": {"sourceTags.name.keyword": "未来科学大奖官网"}}
                ]
            }
        },
        "aggs": {
            "source_tags": {
                "terms": {
                    "field": "sourceTags.name.keyword",
                    "size": 1000
                }
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    
    print("\n一席和未来科学大奖官网的sourceTags统计信息:")
    for bucket in res['aggregations']['source_tags']['buckets']:
        print(f"sourceTags: {bucket['key']}, 数量: {bucket['doc_count']}")
    return res

# 执行统计
# get_source_tags_stats()
# 获取一席和未来科学大奖官网文章的专业统计信息
def get_related_majors_stats():
    query = {
        "size": 0,
        "query": {
            "bool": {
                "should": [
                    {"match": {"sourceTags.name.keyword": "一席"}},
                    {"match": {"sourceTags.name.keyword": "未来科学大奖官网"}}
                ]
            }
        },
        "aggs": {
            "related_majors": {
                "terms": {
                    "field": "relatedMajors.name.keyword",
                    "size": 1000
                }
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    
    result = {
        "专业统计": []
    }
    print(res['aggregations']['related_majors']['buckets'])
    for bucket in res['aggregations']['related_majors']['buckets']:
        result["专业统计"].append({
            "专业名称": bucket['key'].replace("学门:", ""),
            "文章数量": bucket['doc_count']
        })
        print(bucket['key'].replace("学门:", ""))
    import json
    with open('专业统计.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result

# 执行统计
# get_related_majors_stats()



# 获取reviewStatus的统计信息
def get_review_status_stats():
    query = {
        "size": 0,
        "aggs": {
            "review_status": {
                "terms": {
                    "field": "reviewStatus.keyword",
                    "size": 10
                }
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    
    print("审核状态统计:")
    for bucket in res['aggregations']['review_status']['buckets']:
        print(f"状态: {bucket['key']}, 数量: {bucket['doc_count']}")
    return res

# 执行统计
# get_review_status_stats()



# 查找标题中包含"减重神药"的文章
# def find_articles_with_weight_loss_drug():
#     query = {
#         "query": {
#             "match_phrase": {
#                 "title": "中风发病"
#             }
#         }
#     }
    
#     res = es.search(index="shijie", body=query)
    
#     print("包含'中风发病'的文章:")
#     for hit in res['hits']['hits']:
#         print(f"标题: {hit['_source']['title']}")
#         print(f"审核状态: {hit['_source']['reviewStatus']}")
#         print("---")
#     return res

# # 执行查询
# find_articles_with_weight_loss_drug()



# 获取所有审核状态为reviewed的文章
def get_reviewed_articles():
    query = {
        "query": {
            "match": {
                "reviewStatus.keyword": "reviewed"
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    
    print("审核状态为reviewed的文章:")
    print(f"共找到 {res['hits']['total']['value']} 篇文章")
    for index, hit in enumerate(res['hits']['hits']):
        print(f"标题: {hit['_source']['title']}")
        # print(f"内容长度: {hit['_source']['contentSize']}")
        print("---")
        print(hit['_source'])
        if index > 10:
            break
    return res

# 执行查询
# get_reviewed_articles()


# get_sample_doc("shijie")    

# 根据id查找记录
def find_record_by_id(id):
    query = {
        "query": {
            "match": {
                "id": id
            }
        }
    }
    
    res = es.search(index="shijie", body=query)
    
    print(f"查找id为{id}的记录:")
    if res['hits']['total']['value'] > 0:
        for hit in res['hits']['hits']:
            print(hit['_source'])
    else:
        print("未找到记录")
    return res

# 执行查询
# find_record_by_id("1785157995843600")


# 找5条contentSize大于500000的记录
def find_record_with_large_content_size():
    query = {
        "query": {
            "range": {
                "contentSize": {"lt": 500000}  # contentSize是整数类型,直接比较数值
            }
        },
        "size": 5,
        "sort": [
            {"contentSize": "desc"}  # 按内容大小升序
        ]
    }  
    
    res = es.search(index="shijie", body=query)
    
    print(f"内容长度大于500000的文章:")
    print(f"共找到 {res['hits']['total']['value']} 篇文章")
    for hit in res['hits']['hits']:
        print(f"标题: {hit['_source']['title']}")
        print(f"内容大小: {hit['_source']['contentSize']}")
        print(hit['_source'])
        print("---")
    return res

# 执行查询
find_record_with_large_content_size()



# es_delete_index("shijie")
# es_create_index("shijie")

