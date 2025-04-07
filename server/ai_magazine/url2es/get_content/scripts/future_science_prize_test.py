import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
from lxml import etree
import html2text as ht
import re
from scrapy_scripts.common_utils.qqcloud import qqcloud
from ..tools import htmlToMarkDown, replace_images_url
import os
from bs4 import BeautifulSoup


def is_valid_content_tag(tag):
    # 判断标签是否是可能的内容正文标签
    min_text_length = 50  # 最小文本长度阈值
    max_link_density = 0.35  # 最大链接密度阈值

    text = tag.get_text()
    links = tag.find_all('a')

    return len(text) > min_text_length and len(links) / len(text.split()) < max_link_density



def score_tag(tag):
    # 计算标签的得分，考虑文本长度和标签深度
    text_length = len(tag.get_text(strip=True))
    depth = len(tag.find_parents())
    return text_length / (depth + 1)

def get_document_detail(text, qqclient, source):
    """获取html页面内的正文内容，需要将图片内容转为网络上的路径

    Args:
        text (text): html文本

    Returns:
        str: 转为md的文本
    """
    
    soup = BeautifulSoup(text, 'html.parser')
    
    # 找到可能包含内容正文的标签
    potential_content_tags = soup.find_all(['div', 'article', 'main', 'section', 'body', 'content'])

    # 遍历每个可能的标签，选择包含最多文本的标签作为内容正文标签
    best_content_tag = None
    max_text_length = 0

    for tag in potential_content_tags:
        if is_valid_content_tag(tag):
            text_length = len(tag.get_text())
            if text_length > max_text_length:
                max_text_length = text_length
                best_content_tag = tag

    # 如果找到内容正文标签，则返回标签的HTML内容，否则返回空字符串
    if best_content_tag:
        pass
    else:
        return ""
    
    text2 = htmlToMarkDown(str(best_content_tag))
    text2 = replace_images_url(text2, "http://www.futureprize.org", qqclient)
    return text2


if __name__ == '__main__':
    res = get_content()
    print(res)