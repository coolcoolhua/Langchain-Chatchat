
import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import json
import os
from .qqcloud import *

from bs4 import BeautifulSoup


def get_history_para(html_text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, 'lxml')
    content_div = soup.find('div', class_='J-lemma-content')

    # 定位第二个标题div
    second_title_div = content_div.find_all('div', class_='paraTitle_izoVC level-1_deXHA')[1]

    # 提取从最上层div直到第二个标题div之间的内容
    first_part_content = str(content_div)[:str(content_div).find(str(second_title_div))]
    print(first_part_content)
    return first_part_content



def get_original_html_from_cloud(url):
    target_url = "https://url-to-html-puppeteest-nldb-cjryfuvmdu.cn-hangzhou.fcapp.run"

    payload = json.dumps({
        "url": url
    })
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", target_url, headers=headers, data=payload)
    res = response.json()
    return res["html"]

def control_unit(target_website):
    
    """
    
    获取网页原始数据，并保存到cos

    Returns:
        _type_: _description_
    """
    
    print(target_website , '，正在获取网页原始内容..')
    
    source_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤1.json'
    target_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤2.json'
    
    if not os.path.exists('./server/ai_magazine/saved_data/原始网页数据/' + target_website):
        os.mkdir('./server/ai_magazine/saved_data/原始网页数据/' + target_website)
    
    # 获取前一步获取到的数据
    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))
        
    print('共', len(datas), '条数据')
    
    # 初始化腾讯云文件桶实例
    qqclient = qqcloud(target_website)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
    }
    # cj = browser_cookie3.chrome()
    all_urls = []

    for index,data in enumerate(datas):
        if index % 5 == 0:
            print(target_website,"获取网页原始内容,正在处理第{}个网页".format(index))
        if data['originalCosUrl'] != "":
            print("第{}个网页已经处理过了".format(index),'跳过')
            continue
        url = data['originalUrl']
        try:
            # 动态加载
            response = get_original_html_from_cloud(url)
            res = response
            # 静态加载
            # response = requests.get(url, headers = headers, cookies = cj, timeout = 10)
            # response.encoding = 'utf-8'
            # res = response.text
        except Exception as e:
            print(e)
            print("第{}个网页获取失败".format(index),"跳过")
            continue
        
        if len(res) < 10:
            print("获取html步骤失败,此文章跳过")
            data['wrong_reason'] = "获取html步骤失败,此文章跳过"
            json.dump(datas, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            continue
        
        
        # 如果是百度百科数据，做一个处理，只提取某些内容
        if '院校历史' in source_path:
            res = get_history_para(res)
        local_path = "./server/ai_magazine/saved_data/原始网页数据/"+ target_website + "/" + str(index) + ".txt"
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(res)
            f.close()
            remote_path = "original_web_html/" + data["contentUUID"] + ".txt"
            upload_res = qqclient.upload_file(local_path, remote_path)
            data['originalCosUrl'] = upload_res[1]
            data['data_local_path'] = local_path
        json.dump(datas, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print("第{}个网页处理完成并上传成功".format(index))
        time.sleep(random.randint(1,3))
    json.dump(datas, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False,indent=4)
    
    
if __name__ == "__main__":
    res = get_original_html_from_cloud("https://www.gotopku.cn/index/detail/1311.html")
    print(res)