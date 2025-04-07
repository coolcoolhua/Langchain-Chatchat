import requests
import json
import time
import os 
import re

if os.path.exists('./server/ai_magazine/已上传链接.json'):
    all_urls = json.load(open('./server/ai_magazine/已上传链接.json', 'r', encoding='utf-8'))
else:
    all_urls = []
    
    
def md_text_fix(text):
    new_text = re.sub(r'\*\*(.+?)\*\*', r'**\1** ', text)
    return new_text

def split_into_batches(arr, batch_size):
    batches = []
    num_batches = len(arr) // batch_size
    if len(arr) % batch_size != 0:
        num_batches += 1
        
    for i in range(num_batches):
        start = i * batch_size 
        end = start + batch_size
        batch = arr[start:end]
        batches.append(batch)
        
    return batches
    

def post_lark_api(target_website):
    source_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤8.json'
    datas = json.load(open(source_path))
    token = ""
    for index,data in enumerate(datas):
        try:
            if index% 10 ==0:
                # 获取token
                url = "https://ae-openapi.feishu.cn/auth/v1/appToken"
                payload = "{\"clientId\":\"c_f917192b155c47449b6a\",\"clientSecret\": \"cd4ce275ab1e4e349e942e6d96d1563c\"}"
                headers = {
                "Content-Type": "application/json"
                }
                response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
                # 写入接口
                token = response.json()["data"]['accessToken']
            if data['originalUrl'] in all_urls:
                print('重复url')
                continue
            if 'wrong_reason' in data.keys():
                print(index,'数据有错,不上传')
                continue
            temp = {
                    "record": {
                        "contentUUID": data["contentUUID"],
                        "keywords": data["keywords"],
                        "rawJSON": data["rawJSON"],
                        "title": data['title'],
                        "reviewStatus": 'toBeReview',
                        "contentUrl": data['contentUrl'],
                        "description": data['description'],
                        "originalUrl": data["originalUrl"],
                        "originalCosUrl": data["originalCosUrl"],
                        "sourceTags": data['sourceTags'],
                        "topicTags": data['topicTags'],
                        "sensitive_words": data['sensitive_words'],
                        "relatedMajors": data['relatedMajors'],
                        "relatedUniversities": data['relatedUniversities'],
                        "markdownContent" : md_text_fix(data['md_content']),
                        "pub_time_text": data['pub_time_text'],
                        "rawJSON" : json.dumps(data),
                    }
                }
            
            # # record里所有value为空的key都删除
            for key in list(temp['record'].keys()):
                if temp['record'][key] == '' or temp['record'][key] == [] or temp['record'][key] == {}:
                    del temp['record'][key]
            payload = json.dumps(temp)
            
            headers = {
                "Authorization": token,
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
            }
            url = "https://ae-openapi.feishu.cn/v1/data/namespaces/careerContentManagement__c/objects/careerContent/records"
            response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
            all_urls.append(data['originalUrl'])
            # json.dump(all_urls,open('scrapy_scripts/common_utils/all_url_records.json','w',encoding='utf-8'))
            print(response.text)
            time.sleep(0.5)
        except Exception as e:
            print(e)
            print('出错了')
            continue
        
        
def post_lark_api_batch(target_website):
    source_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤8.json'
    datas = json.load(open(source_path))
    token = ""
    datas = split_into_batches(datas, 10)
    for index,data1 in enumerate(datas):
        if index% 5 ==0:
            # 获取token
            url = "https://ae-openapi.feishu.cn/auth/v1/appToken"
            payload = "{\"clientId\":\"c_f917192b155c47449b6a\",\"clientSecret\": \"cd4ce275ab1e4e349e942e6d96d1563c\"}"
            headers = {
            "Content-Type": "application/json"
            }
            response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
            # 写入接口
            token = response.json()["data"]['accessToken']
        if len(token) < 10:
            print('token获取失败')
            return
        post_records = {"records":[]}
        for data in data1:
            try:
                if data['topicTags']["_id"]!="1785156816663577":
                    continue
                temp = {
                            "contentUUID": data["contentUUID"],
                            "keywords": data["keywords"],
                            "rawJSON": data["rawJSON"],
                            "title": data['title'],
                            "reviewStatus": 'toBeReview',
                            "contentUrl": data['contentUrl'],
                            "description": data['description'],
                            "originalUrl": data["originalUrl"],
                            "originalCosUrl": data["originalCosUrl"],
                            "sourceTags": data['sourceTags'],
                            "topicTags": data['topicTags'],
                            "sensitive_words": data['sensitive_words'],
                            "relatedMajors": data['relatedMajors'],
                            "relatedUniversities": data['relatedUniversities'],
                            "markdownContent" : md_text_fix(data['md_content']),
                            "pub_time_text": data['pub_time_text'],
                            "rawJSON" : json.dumps(data),
                        }
                # # record里所有value为空的key都删除
                for key in list(temp.keys()):
                    if temp[key] == '' or temp[key] == [] or temp[key] == {}:
                        del temp[key]
            except Exception as e:
                print(e)
                print('出错了')
                continue
            post_records["records"].append(temp)
        payload = json.dumps(post_records)
        
        headers = {
            "Authorization": token,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
        }
        url = "https://ae-openapi.feishu.cn/v1/data/namespaces/careerContentManagement__c/objects/careerContent/records_batch"
        response = requests.request("POST", url, headers=headers, data=payload)
        # all_urls.extend([data['originalUrl'] for data in data1])
        # json.dump(all_urls,open('scrapy_scripts/common_utils/all_url_records.json','w',encoding='utf-8'))
        print(response.text)
        time.sleep(0.5)
    
    
if __name__ == '__main__':
    # post_lark_api('未来科学大奖')
    post_lark_api_batch('复旦人物')