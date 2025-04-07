import requests
import json
import requests
from bs4 import BeautifulSoup


def get_document_detail(html_text,target_website):
    """
    
    用云函数获取html页面内的正文内容
    

    Args:
        text (text): html文本

    Returns:
        str: 转为md的文本
    """
    
    
    # 获取token
    url = "https://ae-openapi.feishu.cn/auth/v1/appToken"
    payload = "{\"clientId\":\"c_f917192b155c47449b6a\",\"clientSecret\": \"cd4ce275ab1e4e349e942e6d96d1563c\"}"
    headers = {
    "Content-Type": "application/json"
    }
    response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
    token = response.json()["data"]['accessToken']
    # 写入接口
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }
    if 'yixi.tv' in target_website:
        html_text = json.loads(html_text)['data']['draft']

    
    url = "https://ae-openapi.feishu.cn/api/cloudfunction/v1/namespaces/careerContentManagement__c/invoke/html_to_markdown"
    data = {
        "params":{
            "html": html_text,
            "url": target_website
        }
    }
    # print(data)
    payload = json.dumps(data)
    response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
    response.encoding = 'utf-8'
    try:
        res = response.json()['data']['result']['markdown']
    except Exception as e:
        print(response.text)
        print(e)
        res = ""
    return res


if __name__ == '__main__':
    text = ''.join(open('saved_data/原始网页数据/北京大学招生办/0.txt', 'r', encoding='utf-8').readlines())
    res = get_document_detail(text, "https://www.pku.edu.cn/xxgk/xxgk.html")
    print(res)