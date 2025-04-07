import requests
import json
import requests
import browser_cookie3
from lxml import html
from lxml import etree
import time
import random
import json
import os
from .qqcloud import qqcloud
from .tools import save_remote_md

def get_details(target_website, html_text, qqclient, source):

    from .general_scrapy_new import get_document_detail
    return get_document_detail(html_text, source)


def control_unit(target_website):
    """

    获取网页markdown，并保存到cos

    Returns:
        _type_: _description_
    """

    print(target_website, "，正在提取markdown内容...")

    source_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤4.json"
    )
    target_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤5.json"
    )

    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))

    print("共", len(datas), "条数据")

    # 初始化腾讯云文件桶实例
    qqclient = qqcloud(target_website)

    for index, data in enumerate(datas):
        # print(data['title'])
        if index % 5 == 0:
            print("正在处理第", index, "条数据")
        # 断点传输
        if len(data["contentUrl"]) > 0:
            print("第{}个网页已获取markdown".format(index), "跳过")
            continue
        
        if "data_local_path" not in data:
            print("获取html步骤失败,此文章跳过")
            data['wrong_reason'] = "数据未存到本地，跳过"
            json.dump(
                datas,
                open(target_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4,
            )
            continue
        else:
            if len(data["data_local_path"]) < 10:
                print("获取html步骤失败,此文章跳过")
                data['wrong_reason'] = "数据未存到本地，跳过"
                json.dump(
                    datas,
                    open(target_path, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=4,
                )
                continue
        
        html_text = "".join(
            open(data["data_local_path"], "r", encoding="utf-8").readlines()
        )
        
        if len(html_text) < 10:
            print("html文本异常，跳过")
            data['wrong_reason'] = "html文本异常，跳过"
            json.dump(
                datas,
                open(target_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4,
            )
            continue
        
        print(data["originalUrl"],data["title"])
        md_text = get_details(target_website, html_text, qqclient, data["originalUrl"])
        if len(md_text) < 10:
            print("未获取到markdown文本，提取失败，跳过")
            data['wrong_reason'] = "未获取到markdown文本，提取失败，跳过"
            data["md_content"] = ""
            data["contentUrl"] = ""
            continue
        md_text = md_text.replace("shijiesys-1256650073.cos", "https://shijiesys-1256650073.cos")
        # print(md_text)
        data["md_content"] = md_text
        data["contentUrl"] = save_remote_md(md_text, "", qqclient, data["contentUUID"])
        print(data["contentUrl"])
        json.dump(
            datas,
            open(target_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
    json.dump(
        datas, open(target_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4
    )
