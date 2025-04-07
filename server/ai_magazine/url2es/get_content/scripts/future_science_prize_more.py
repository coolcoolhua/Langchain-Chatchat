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

def get_document_detail(text, qqclient, source):
    """获取html页面内的正文内容，需要将图片内容转为网络上的路径

    Args:
        text (text): html文本

    Returns:
        str: 转为md的文本
    """
    text_html = etree.HTML(text)
    content_etree = text_html.xpath("//div[@class='content']")
    if not content_etree:
        return ""
    res = etree.tostring(content_etree[0], encoding="unicode").strip()
    text2 = htmlToMarkDown(res)
    text2 = replace_images_url(text2, "http://www.futureprize.org", qqclient)
    return text2


if __name__ == '__main__':
    res = get_content()
    print(res)