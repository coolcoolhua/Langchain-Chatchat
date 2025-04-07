

import io
import contextlib
import importlib
import json
import sys

import importlib
from .utils.basic_info_format import get_basic_info
from .utils.qqcloud import qqcloud
from .utils.get_lark_tag_source import get_lark_tag_source
from .utils.get_lark_tag_school import get_lark_tag_school
from .utils.get_lark_tag_major import get_lark_tag_major
import uuid
from elasticsearch import Elasticsearch
from fastapi import Body
from fastapi.responses import JSONResponse
from typing import *


def url2es(
    url: str = Body(..., description="索引名称", examples=["shijie"]),
    mode: str = Body(..., description="操作名", examples=["添加数据"])
):
    ret = {
        "status": "",
        "response" : ""
    }
    
        
    target_website = '临时爬取内容'
    target_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤1.json'

    # 基本信息
    basic_info = get_basic_info()
    res = []


    basic_info['description'] = "无"
    basic_info['originalUrl'] = url
    basic_info["contentUUID"] = str(uuid.uuid1()).replace("-","")
    basic_info['auditStatus'] = 'option_dc305ba43bd'

    res.append(basic_info)
    json.dump(res, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False,indent=4)
    
    
    from .get_original_data import control_unit as get_original_data
    # 存url内容到本地
    get_original_data(target_website)

    # # 3. 获取元数据
    # # """
    
    from .get_meta_data import control_unit as get_meta_data
    get_meta_data(target_website)

    # # """
    # # 4. 元数据过滤
    # # """
    
    from .get_meta_filter import control_unit as get_meta_filter
    get_meta_filter(target_website)

    # 获取markdown
    
    from .get_content import control_unit as get_content
    get_content(target_website)


    # 根据markdown文本获取标题
    target_path1 = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤5.json'
    data = json.load(open(target_path1, 'r', encoding='utf-8'))
    data = data[0]
    md_text = data['md_content']
    temp = md_text.split('\n\n')
    get_title = False   
    for t in temp[:20]:
        # 如果有标题就用标题，没有标题就用第一行
        if "###" in t:
            data['title'] = t.replace("###", "").strip()
            get_title = True
            break
    if get_title == False:
        data['title'] = temp[0].strip()
    data = [data]
    json.dump(data, open(target_path1, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


    # # """
    # # 6.获取敏感词
    # # """
    
    # from .utils.get_sensitive_words import get_sensitive_words
    
    # get_sensitive_words(target_website)

    # # # """
    # # # 7.打标签
    # # # """
    
    # from .utils.get_llm_tags_content import get_llm_tags as get_llm_tags_content
    # get_llm_tags_content('内容',target_website)
    
    from .utils.get_llm_tags_major import get_llm_tags as get_llm_tags_major
    get_llm_tags_major('专业',target_website)

    # """
    # 8.写入飞书标签
    # """
    
    
    from .utils.post_lark_api import post_lark_api
    post_lark_api(target_website)



    
    return JSONResponse(ret)



