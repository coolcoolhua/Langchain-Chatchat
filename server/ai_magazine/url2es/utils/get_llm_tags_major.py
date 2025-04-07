from langchain.utilities import BingSearchAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.docstore.document import Document
import requests
import json
import requests
import browser_cookie3
from lxml import html
import time
import random
import os
from .get_lark_tag_major import *
from .get_lark_tag_school import *
import thulac

from langchain.utilities import DuckDuckGoSearchAPIWrapper
thu = thulac.thulac()

def duckduckgo_search(text, result_len=5):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)

def get_keywords(data):
    text = data['title']
    if text == None:
        return ""
    else:
        print(text)
    # 进行分词并获取词性标注
    result = thu.cut(text, text=True)
    print(result)
    # 分词提取人名
    names = []
    current_name = ""
    for word in result.split(' '):
        if 'np' in word or 'ni' in word :  # 'np' 表示人名
            current_name += word.split('/')[0]
        else:
            if current_name:
                names.append(current_name.split('_')[0])
                current_name = ""
    # 处理最后一个人名
    if current_name:
        names.append(current_name.split('_')[0])
    
    # 如果有相关院校就加上
    if data['relatedUniversities']!=[]:
        if type(data['relatedUniversities']) == dict:
            data['relatedUniversities'] = [data['relatedUniversities']]
        for t in data['relatedUniversities']:
            print(t)
            names.append(get_lark_tag_school_name(t["_id"]))
        
    # 加上
    print(names)
    print("最终搜索query:",','.join(names) +','+ data['title'])
    
    
    return ','.join(names) +','+ data['title']



majors = {
    "哲学": ["哲学类"],
    "经济学": ["经济学类", "财政学类", "金融学类", "经济与贸易类"],
    "法学": ["法学类", "政治学类", "社会学类", "民族学类", "马克思主义理论类", "公安学类"],
    "教育学": ["教育学类", "体育学类"],
    "文学": ["中国语言文学类", "外国语言文学类", "新闻传播学类"],
    "历史学": ["历史学类"],
    "理学": [
        "数学类",
        "物理学类",
        "化学类",
        "天文学类",
        "地理科学类",
        "大气科学类",
        "海洋科学类",
        "地球物理学类",
        "地质学类",
        "生物科学类",
        "心理学类",
        "统计学类",
    ],
    "工学": [
        "力学类",
        "机械类",
        "仪器类",
        "材料类",
        "能源动力类",
        "电气类",
        "电子信息类",
        "自动化类",
        "计算机类",
        "土木类",
        "水利类",
        "测绘类",
        "化工与制药类",
        "地质类",
        "矿业类",
        "纺织类",
        "轻工类",
        "交通运输类",
        "海洋工程类",
        "航空航天类",
        "兵器类",
        "核工程类",
        "农业工程类",
        "林业工程类",
        "环境科学与工程类",
        "生物医学工程类",
        "食品科学与工程类",
        "建筑类",
        "安全科学与工程类",
        "生物工程类",
        "公安技术类",
        "交叉工程类",
    ],
    "农学": ["植物生产类", "自然保护与环境生态类", "动物生产类", "动物医学类", "林学类", "水产类", "草学类"],
    "医学": [
        "基础医学类",
        "临床医学类",
        "口腔医学类",
        "公共卫生与预防医学类",
        "中医学类",
        "中西医结合类",
        "药学类",
        "中药学类",
        "法医学类",
        "医学技术类",
        "护理学类",
    ],
    "管理学": [
        "管理科学与工程类",
        "工商管理类",
        "农业经济管理类",
        "公共管理类",
        "图书情报与档案管理类",
        "物流管理与工程类",
        "工业工程类",
        "电子商务类",
        "旅游管理类",
        "质量管理工程类",
        "监狱学类",
    ],
    "艺术学": ["艺术学理论类", "音乐与舞蹈学类", "戏剧与影视学类", "美术学类", "设计学类"],
}

tag_list = ["哲学", "经济学", "法学", "教育学", "文学", "历史学", "管理学", "理学", "工学", "农学", "医学", "艺术学"]

# 把major里的所有值放到一个list里
all_major_list = []
for key in majors.keys():
    all_major_list += majors[key]


def duckduckgo_search(text, result_len=5):
    print("开始搜索")
    search = DuckDuckGoSearchAPIWrapper()
    search.region = 'cn-zh'
    return search.results(text, result_len)


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"] if "snippet" in result.keys() else "",
            metadata={
                "source": result["link"] if "link" in result.keys() else "",
                "filename": result["title"] if "title" in result.keys() else "",
            },
        )
        docs.append(doc)
    return docs


def server_tag(tag_type="1级专业", text="", rag_content=""):

    url = "http://0.0.0.0:6006/chat/context_chat_tag"
    if tag_type == "1级专业":
        query = (
            text + rag_content[:1000] + "。根据已知信息和文章相关信息，判断文章属于以下哪个专业，可选专业有：哲学、经济学、法学、教育学、文学、历史学、理学、工学、农学、医学、管理学、艺术学。如果和专业不相关(比如是院校类信息等)，就说无法判断。要严格按照呢可选专业内的名字来回答。"
        )
    elif tag_type == "2级专业":
        major = tag_type.split("_")[1]
        query = (
            text + rag_content[:1000] +  "。根据已知信息和文章相关信息，判断文章属于以下哪个专业。可选专业有：" + ",".join(majors[major]) + '。如果和专业不相关(比如是院校类信息等)，就说无法判断。要严格按照呢可选专业内的名字来回答。'
        )
    else:
        query = (
            text + rag_content[:1000] + "。根据已知信息和文章相关信息，判断文章属于以下哪个专业。可选专业有：" + ",".join(all_major_list) + '。如果和专业不相关(比如是院校类信息等)，就说无法判断。要严格按照呢可选专业内的名字来回答。'
        )
    payload = {
        "query": query,
        "context": text,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload).encode('utf-8'), headers=headers, timeout=20)
    print('接口返回内容',response.text)
    llm_tag = response.json()["answer"]
    # 需要对返回的数据进行处理
    if tag_type == "1级专业":
        for tag in tag_list:
            if tag in llm_tag:
                print("命中tag",tag)
                return tag
        return "工学"

    if tag_type.startswith("2级专业"):
        major_prefix = tag_type.split("_")[1]
        candidates = majors[major_prefix]
        # 防止因为pipeline导致的二级标签错误
        for tag in all_major_list:
            if tag in llm_tag:
                return tag
        # 二级标签没有命中的，兜底
        for tag in candidates:
            if tag in llm_tag:
                return tag
        return 
    
    if tag_type.startswith("3级专业"):
        candidates = all_major_list
        # 防止因为pipeline导致的二级标签错误
        for tag in all_major_list:
            if tag in llm_tag:
                return tag
        # 二级标签没有命中的，兜底
        for tag in candidates:
            if tag in llm_tag:
                return tag
        return 


def get_llm_tags(tag_type, target_website):
    """

    未来科学大奖

    Returns:
        _type_: _description_
    """

    print(target_website + "，正在通过LLM获取专业标签...")

    source_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤7.json"
    )
    target_path = (
        "./server/ai_magazine/saved_data/中间结果/" + target_website + "/" + target_website + "_步骤8.json"
    )

    if os.path.exists(target_path):
        datas = json.load(open(target_path))
    else:
        datas = json.load(open(source_path))

    for index, data in enumerate(datas):
        print("第",index,"个页面")
        # 断点传输
        if len(data["relatedMajors"]) != 0:
            print("第{}个网页已经处理过了".format(index), "跳过")
            continue
        
        if 'md_content' not in data.keys() or 'wrong_reason' in data.keys():
            print("第{}篇文章前续步骤有错，无法获取标签".format(index))
            continue
        
        if len(data['topicTags']) >0 and (data['topicTags']["_id"] == "1785156816478339" or data['topicTags']["_id"] == "1785156816663593"):
            print("第{}篇文章是院校类信息，不需要获取标签".format(index))
            data["relatedMajors"] = []
            json.dump(
                datas,
                open(
                    target_path,
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
            continue
        # 通过LLM获取标签
        search_keywords = get_keywords(data)
        # print(search_keywords)
        try:
            # docs = search_result2docs(duckduckgo_search(search_keywords))
            docs = []
            print("搜索结束")
            rag_content = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print("第{}篇文章搜索出错".format(index),e)
            rag_content = ""
        # print(rag_content)
        try:
            first_res_title = server_tag(
                "1级专业", "文章相关信息：标题: " + data["title"] + "内容: " + data["md_content"][:500], rag_content
            )
            second_res_title = server_tag(
                "2级专业" + "_" + first_res_title,
                "文章相关信息：标题: " + data["title"] + "内容: " + data["md_content"][:500], rag_content
            )
            third_res_title = server_tag(
                "3级专业",
                "文章相关信息：标题: " + data["title"] + "内容: " + data["md_content"][:300], rag_content
            )
            print("1级专业：", first_res_title, "2级专业：", second_res_title, "3级专业：", third_res_title)
            data["relatedMajors"] = []
            if first_res_title != None:
                data["relatedMajors"].append(get_lark_tag_major(first_res_title))
            if second_res_title != None and second_res_title != first_res_title:
                data["relatedMajors"].append(get_lark_tag_major(second_res_title))
            if third_res_title != None and third_res_title != second_res_title and third_res_title != first_res_title:
                data["relatedMajors"].append(get_lark_tag_major(third_res_title))
            print("最终标签", data["relatedMajors"])
            print(index, "/", len(datas) , data["title"])
            json.dump(
                datas,
                open(
                    target_path,
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
        except Exception as e:
            print("第{}篇文章LLM出错".format(index), e)
            data["relatedMajors"] = []
            json.dump(
                datas,
                open(
                    target_path,
                    "w",
                    encoding="utf-8",
                ),
                ensure_ascii=False,
                indent=4,
            )
            continue

    print(target_website, "标签获取完毕！")
    return


if __name__ == "__main__":
    res = get_llm_tags("专业", "未来科学大奖_更多")
    print(res)
