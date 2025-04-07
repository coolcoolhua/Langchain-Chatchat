from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging
import pypinyin

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.ERROR, stream=sys.stdout)

    
def hp(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


class qqcloud:
    def __init__(self, website_name):
        self.secret_id = "AKIDPdbLp41mGVVHp8g1cx2rSr0tVryoLinc"
        self.secret_key = "h9ZmkUzFH4dx2uYx72kDQM3TdW8UaP32"
        self.region = "ap-beijing"
        self.bucket = "shijiesys-1256650073"
        self.client = CosS3Client(CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key))
        self.location_base = "https://image.sjsys.shijieu.cn/"
        self.remote_path_prefix = "scrapy_resources/" + self.name_trans(website_name) + "/"
    
    def upload_file(self, local_path, remote_path):
        # 上传文件
        config = CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key)
        client = CosS3Client(config)
        response = client.put_object_from_local_file(
            Bucket = self.bucket,
            LocalFilePath = local_path,
            Key = self.remote_path_prefix + remote_path,
        )
        return response , self.location_base + self.remote_path_prefix + remote_path
    
    def download_file(self, local_path, remote_path):
        # 下载文件
        config = CosConfig(Region=self.region, SecretId=self.secret_id, SecretKey=self.secret_key)
        client = CosS3Client(config)
        response = client.get_object(
            Bucket=self.bucket,
            Key=remote_path,
        )
        response['Body'].get_stream_to_file(local_path)
        return response
    
    def name_trans(self, name):
        if name == '临时爬取内容':
            return 'temp_website'
        elif name== '未来科学大奖':
            return 'fsp'
        else:
            return hp(name)
        
if __name__ == '__main__':
    qqcloud = qqcloud('fsp')
    local_path = '爬虫流程/saved_data/中间结果/未来科学大奖.json'
    remote_path = '未来科学大奖.json'
    qqcloud.upload_file(local_path, remote_path)