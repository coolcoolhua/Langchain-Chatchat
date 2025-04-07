
import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import json
import os


def control_unit(target_website):
    
    """
    
    获取网页原始数据，并保存到cos

    Returns:
        _type_: _description_
    """
    
    print(target_website , '，正在进行获取元数据..')
    
    source_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤2.json'
    target_path = './server/ai_magazine/saved_data/中间结果/' + target_website + '/' + target_website + '_步骤3.json'
    
    if not os.path.exists('./server/ai_magazine/saved_data/原始网页数据/' + target_website):
        os.mkdir('./server/ai_magazine/saved_data/原始网页数据/' + target_website)
    
    # 获取前一步获取到的数据
    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))
        
    print('共', len(datas), '条数据')
    json.dump(datas, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False,indent=4)