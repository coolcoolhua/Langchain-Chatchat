import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import os
from .get_lark_tag_content import *


def server_tag(tag_type="内容", text = ""):
    url = "http://0.0.0.0:6006/chat/get_content_tags"

    payload = {
        "query": tag_type,
        "content": text,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload).encode('utf-8'), headers=headers, timeout=20)
    print(response.text)
    llm_tag = response.text.strip().replace(' ','')
    print("大模型答案：", llm_tag)
    # 需要对返回的数据进行处理
    if tag_type == "内容":
        tag_list = ["专业重点扫盲", "专业杰出人物", "专业前沿资讯", "院校历史", "院校政策"]
    for tag in tag_list:
        if tag in llm_tag:
            print("命中tag",tag)
            return tag
    print("未命中tag,返回默认值，专业杰出人物")
    return "专业杰出人物"


def get_llm_tags(tag_type, target_website):
    """

    未来科学大奖

    Returns:
        _type_: _description_
    """

    print(target_website + "，正在通过LLM获取内容标签...")

    source_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤6.json"
    )
    target_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤7.json"
    )

    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))
    
    if '招生办' in source_path or '院校历史' in source_path:
        # 招生政策已经处理过了 直接退出
        json.dump(
            datas,
            open(
                target_path,
                'w',
                encoding="utf-8",
            ),
            ensure_ascii=False,
            indent=4,
        )
        return 
        
    for index,data in enumerate(datas):
        # 断点传输
        if len(data["topicTags"]) != 0:
            continue
        # 判断前续步骤是否有错
        if 'md_content' not in data.keys() or 'wrong_reason' in data.keys():
            print("第{}篇文章前续步骤有错，无法获取标签".format(index))
            continue
        
        # 通过LLM获取标签
        try:
            tag = server_tag(tag_type, '标题：' + data["title"] + '\n正文: ' + data["md_content"][:300])
            data["topicTags"]=get_lark_tag_content(tag)
            print(index,"/",len(datas),data["title"], data["topicTags"], "命中tag", tag)
            json.dump(
                datas,
                open(
                    target_path,
                    'w',
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
        except Exception as e:
            print(e)
            print("第{}篇文章获取topic出错".format(index))
            continue

    print(target_website, "标签获取完毕！")

    return


if __name__ == "__main__":
    res = get_llm_tags("内容", "未来科学大奖_更多")
    print(res)
