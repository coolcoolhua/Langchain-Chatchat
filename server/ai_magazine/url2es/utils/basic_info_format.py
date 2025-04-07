
def get_basic_info():
    # 定义所有需要爬取的内容，分发给所有爬取模块
    basic_info = {
        # 内容唯一标识符
        "contentUUID" : "",
        # 内容标题
        "title": "",
        # 审核人员
        "auditors": {},
        # 来源标签
        "sourceTags": {},
        # 关联大学
        "relatedUniversities": [],
        # 内容地址，存放md文件的位置
        "contentUrl": "",
        # 内容描述
        "description": "",
        # 审核状态
        "reviewStatus": "",
        # 原始内容地址
        "originalUrl": "",
        # 原始内容cos地址存档
        "originalCosUrl": "",
        # 搜索关键词
        "keywords": "",
        # 原始内容格式
        "originalContentFormat": {},
        # 原始JSON
        "rawJSON": "",
        # 敏感词内容
        "sensitive_words": [],
        # 内容原文件
        "contentAttachment": [],
        # 内容预览地址
        "previewUrl": "",
        # 审核结论
        "auditComment": "",
        # 主题标签
        "topicTags": {},
        # 其他标签
        "otherTags": [],
        # 相关专业
        "relatedMajors": [],
        # 发布时间
        "pub_time_text": "",
    }
    
    return basic_info