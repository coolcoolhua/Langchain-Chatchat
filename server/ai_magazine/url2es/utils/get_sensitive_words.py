import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import os

def get_sensitive_words(website_name):
    
    """
    
    列出所有敏感词
    
    Returns:
        _type_: _description_
    """
    
    print(website_name + '，正在获取敏感词...')
    source_path = './server/ai_magazine/saved_data/中间结果/' + website_name + '/' + website_name + '_步骤5.json'
    target_path = './server/ai_magazine/saved_data/中间结果/' + website_name + '/' + website_name + '_步骤6.json'
    
    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))
    
    bad_words = [t.strip() for t in open('./server/ai_magazine/url2es/utils/badwords.txt', 'r', encoding='utf-8').readlines() if t.strip()!='']
    for index,data in enumerate(datas):
        # 断点传输
        if 'sensitive_checked' in data:
            print('第{}个网页已经处理过了'.format(index), '跳过')
            continue
        if 'md_content' not in data:
            print('第{}个网页没有md_content'.format(index), '跳过')
            continue
        data['sensitive_words'] = []
        for word in bad_words:
            # if word in data['title'] or word in data['description'] or word in data['md_content'] or word in data['content']:
            if word in data['md_content']:
                # 找到word所有的index
                indexes = [i for i, x in enumerate(data['md_content']) if x == word]
                for index in indexes:
                    data['sensitive_words'].append({
                        'word': word,
                        'context': data['md_content'][index-10:index+10],
                    })
        data['sensitive_checked'] = True
        json.dump(datas, open(target_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    print(website_name + '，敏感词获取完毕！')
    
    return 





if __name__ == '__main__':
    res = get_sensitive_words('未来科学大奖')
    print(res)