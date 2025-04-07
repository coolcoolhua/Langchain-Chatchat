# import html2text as ht
import re 
import urllib.request
import time
import os 


def replace_images_url(text,target_website, qqclient):
     # 正则匹配，提取所有以图片格式结尾的链接
    if "https://mmbiz.qpic.cn" in text:
        pattern = re.compile(r'\((https?://[^\s]+)\)')
        img_urls = pattern.findall(text)
        old_img_urls = img_urls.copy()
    else:
        pattern = re.compile(r'\((.*?)\.(jpg|png|svg|gif|bmp|jpeg)\)')
        img_urls = pattern.findall(text)
        old_img_urls = [(t[0]+ '.' + t[1]) for t in img_urls]
        img_urls = [(t[0]+ '.' + t[1]).replace("\\","") for t in img_urls]
    print("图片链接",img_urls)
    text2 = text
    for index,img_url in enumerate(img_urls):
        if img_url.startswith('data:image'):
            # base64 不处理
            continue
        if img_url.startswith('http'):
            # 说明是完整链接，不用管
            true_path = img_url
            print("完整链接",true_path)
        else:
            # 说明是相对链接，需要加上前缀
            true_path = target_website + img_url.replace('../','/').replace('./','/')
            print("相对链接",true_path)
            # 下载true_path的图片到本地,并返回本地路径,只需要一个临时的文件名即可
        local_path = download_image(true_path)            
        img_format= get_format(img_url)
        # 根据时间戳生成一个随机的文件名
        remote_name = 'images/' +  str(int(time.time())) + '.' + img_format
        if local_path!="":
            # 上传到腾讯云
            upload_path = qqclient.upload_file(local_path, remote_name)
            # print("最终地址",upload_path)
            text2 = text2.replace(old_img_urls[index], upload_path[1])
            # 删除本地文件
            os.remove(local_path)
        else:
            text2 = text2
            
    # print("here",text2)
    return text2

def save_remote_md(text, target_website, qqclient, contentUUID):
    with open('temp.md', 'w', encoding='utf-8') as f:
        f.write(text)
        f.close()
    if os.path.exists('temp.md'):
        # 上传到腾讯云
        remote_name = 'markdowns/' + contentUUID + '.md'
        upload_path = qqclient.upload_file('temp.md', remote_name)
        # 删除本地文件
        if os.path.exists('temp.md'):
            os.remove('temp.md')
    return upload_path[1]
    


def download_image(url):
    # 下载图片到本地
    # 生成一个随机的文件名
    
    img_format = get_format(url)
    time_stamp = str(time.time())
    filename = time_stamp + '.' + img_format
    # 下载图片
    try:
        urllib.request.urlretrieve(url, filename)
    except:
        print("下载图片失败",url)
        return ""
    return filename


def get_format(url):
    
    temp = str.capitalize(url)
    # 获取图片类型
    if temp.endswith('JPG'):
        return 'jpg'
    elif temp.endswith('JPEG'):
        return 'jpeg'
    elif temp.endswith('PNG'):
        return 'png'
    elif temp.endswith('GIF'):
        return 'gif'
    else:
        return 'png'
    