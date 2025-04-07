from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
from fastapi import Body, Request
import aiohttp
import asyncio
from server.chat.utils import History
import elasticsearch

from webui_pages.utils import *
api = ApiRequest(base_url="http://0.0.0.0:6006")
import time
import re

es = elasticsearch.Elasticsearch()


def history_preprocess(history):
        # 获取最新一条数据里的data字段
    # print("history=",history)
    data = ""
    final_string = ''
    for his in history:
        if his['type'] == 10:
            data = his
            break
       
    extract_ret = {
        "province": None,
        "year": None,
        "intentionSpecialtyList": [],
        "intentionScoreline": [],
        "intentionProvinces": [],
        "tags": [],
        "intentionUniversityList": [],
        "intentionSubjects": []
    }
    print("data=",data)
    if data == "":
        return final_string
    else:
        data = json.loads(data['content'])
    province = data.get('province','')
    year = data.get('year','')
    intentionSpecialtyList = data.get('intentionSpecialtyList',[])
    intentionScoreline = data.get('intentionScoreline',[])
    intentionProvinces = data.get('intentionProvinces',[])
    tags = data.get('tags',[])
    intentionUniversityList = data.get('intentionUniversityList',[])
    intentionSubjects = data.get('intentionSubjects',[])
    print("province=",province)
    print("year=",year)
    print("intentionSpecialtyList=",intentionSpecialtyList)
    print("intentionScoreline=",intentionScoreline)
    print("intentionProvinces=",intentionProvinces)
    print("tags=",tags)
    print("intentionUniversityList=",intentionUniversityList)
    print("intentionSubjects=",intentionSubjects)
    
    if province:
        extract_ret["province"] = province
    if year:
        extract_ret["year"] = year
    if intentionSpecialtyList:
        extract_ret["intentionSpecialtyList"] = intentionSpecialtyList
    if intentionScoreline:
        extract_ret["intentionScoreline"] = intentionScoreline
    if intentionProvinces:
        extract_ret["intentionProvinces"] = intentionProvinces
    if tags:
        extract_ret["tags"] = tags
    if intentionUniversityList:
        extract_ret["intentionUniversityList"] = intentionUniversityList
    if intentionSubjects:
        extract_ret["intentionSubjects"] = intentionSubjects
    print("extract_ret=",extract_ret)
    return extract_ret
        

def extract_info_sep(info,final_data,query,operation_judge):
    if operation_judge != "":
        if "增加" in operation_judge:
            operation_judge = "增加"
        if "删除" in operation_judge:
            operation_judge = "删除"
        if "修改" in operation_judge:
            operation_judge = "修改"
    print("最终操作",operation_judge)
    wrong_flag = False
    if info[1] == 'career_locate':
        try:
            data = eval(info[0])
            print("career_locate=",data)
            province = data
            if operation_judge == "增加":
                final_data['province'] = locate_province_validation(province)
            if operation_judge == "删除":
                final_data['province'] = None
            if operation_judge == "修改":
                final_data['province'] = locate_province_validation(province)
        except Exception as e:
            print("province validation error is", e )
            final_data['province'] = None
            wrong_flag = True
    if info[1] == 'career_year':
        try:
            data = eval(info[0])
            print("career_year= ",data)
            year = data
            if operation_judge == "增加":
                final_data['year'] = year_validation(year)
            if operation_judge == "删除":
                final_data['year'] = None
            if operation_judge == "修改":
                final_data['year'] = year_validation(year)
            # final_data['year'] = year_validation(year)
        except Exception as e:
            print("year validation error is", e )
            final_data['year'] = None
            wrong_flag = True
            
    if info[1] == 'career_major':
        try:
            data = eval(info[0])
            print("check career_major =",data)
            majors = data
            if operation_judge == "增加":
                final_data['intentionSpecialtyList'].extend(major_validation(majors))
            if operation_judge == "删除":
                final_data['intentionSpecialtyList'] = []
            if operation_judge == "修改":
                final_data['intentionSpecialtyList'] = major_validation(majors)
            # final_data['intentionSpecialtyList'] = major_validation(majors)
        except Exception as e:
            print("major validation error is", e )
            final_data['intentionSpecialtyList'] = []
            wrong_flag = True
            
    if info[1] == 'career_school_score':
        try:
            data = eval(info[0])
            print("check career_school_score =",data)
            scores = data
            if operation_judge == "增加":
                final_data['intentionScoreline']=score_validation(scores,query)
            if operation_judge == "删除":
                final_data['intentionScoreline'] = []
            if operation_judge == "修改":
                final_data['intentionScoreline'] = score_validation(scores,query)
            # final_data['intentionScoreline'] = score_validation(scores,query)
        except Exception as e:
            print("score validation error is", e )
            final_data['intentionScoreline'] = []
            wrong_flag = True
            
    if info[1] == 'career_school_province':
        try:
            data = eval(info[0])
            print("check career_school_province=",data)
            school_provinces = data
            if operation_judge == "增加":
                final_data['intentionProvinces'].extend(school_province_validation(school_provinces))
            if operation_judge == "删除":
                final_data['intentionProvinces'] = []
            if operation_judge == "修改":
                final_data['intentionProvinces'] = school_province_validation(school_provinces)
            # final_data['intentionProvinces'] = school_province_validation(school_provinces)
        except Exception as e:
            print("school province validation error is", e )
            final_data['intentionProvinces'] = []
            wrong_flag = True
            
    if info[1] == 'career_school_tag':
        try:
            data = eval(info[0])
            print("career_school_tag=",data)
            school_type = data
            if operation_judge == "增加":
                final_data['tags'].extend(school_type_validation(school_type))
            if operation_judge == "删除":
                final_data['tags'] = []
            if operation_judge == "修改":
                final_data['tags'] = school_type_validation(school_type)
            # final_data['tags'] = school_type_validation(school_type)
        except Exception as e:
            print("school type validation error is", e )
            final_data['tags'] = []
            wrong_flag = True
            
    if info[1] == 'career_schools':
        try:
            data = eval(info[0])
            print("career_schools=",data)
            schools = data
            if operation_judge == "增加":
                final_data['intentionUniversityList'].extend(school_validation(schools))
            if operation_judge == "删除":
                final_data['intentionUniversityList'] = []
            if operation_judge == "修改":
                final_data['intentionUniversityList'] = school_validation(schools)
            
            # final_data['intentionUniversityList'] = school_validation(schools)
        except Exception as e:
            print("school validation error is", e )
            final_data['intentionUniversityList'] = []
            wrong_flag = True
            
    if info[1] == 'career_selection':
        try:
            data = eval(info[0])
            print("career_selection=",data)
            subject_combinations = data
            if operation_judge == "增加":
                final_data['intentionSubjects'].extend(intention_subjects_validate(subject_combinations))
            if operation_judge == "删除":
                final_data['intentionSubjects'] = []
            if operation_judge == "修改":
                final_data['intentionSubjects'] = intention_subjects_validate(subject_combinations)
            
            # final_data['intentionSubjects'] = intention_subjects_validate(subject_combinations)
        except Exception as e:
            print("subject combination validation error is", e )
            final_data['intentionSubjects'] = []
            wrong_flag = True
            
    return final_data, wrong_flag


def intention_subjects_validate(subject_combinations):
    """
    
    检查选课组合的合法性

    Args:
        subject_combinations (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def subject_reformat(combinations):
        return combinations.replace('物理','物').replace('历史','史').replace('政治','政').replace('地理','地').replace('化学','化').replace('生物','生').replace('技术','技')
    
    subject2id = {
        '物理': "25",
        '历史': "29",
        '政治': "31",
        '地理': "30",
        '化学': "26",
        '生物': "27",
        '技术': "28",
        '物': "25",
        '史': "29",
        '政': "31",
        '地': "30",
        '化': "26",
        '生': "27",
        '技': "28"
    }
    
    # 确保subject_combinations是一个list
    if type(subject_combinations) == str:
        subject_combinations = [subject_combinations] 
    
    final_combinations = []
    full_names = ["物理","历史","政治","地理","化学","生物","技术"]
    simple_names = ["物","史","政","地","化","生","技"]
    
    for combination in subject_combinations:
        combination = subject_reformat(combination)
        if combination == "未提及":
            continue
        
        if len(combination) >0:
            final_subjects = []
            for subject in combination:
                if subject not in simple_names:
                    continue
                final_subjects.append(subject2id[subject])
            if final_subjects not in final_combinations:
                final_combinations.append(final_subjects)
        
                

    final_combinations = [t for t in final_combinations if len(t) > 0]
    return final_combinations

def lcs_length(s1, s2):
    """计算两个字符串s1和s2的最长公共子序列长度"""
    m, n = len(s1), len(s2)
    # 创建一个二维数组存储中间结果
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def score_validation(scores,query):
    """
    
    分数的合法性检查

    Args:
        scores (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    branch2id = {
        "不限": 0,
        "文科": 10,
        "理科": 20,
        "物理组": 30,
        "历史组": 40,
        "综合类": 50
    }
    
    return_format = {
        "branch": 0,
        "minScore": 0,
        "minRank": 0,
        "maxScore": 0,
        "maxRank": 0
    }
    
    if type(scores) != list:
        scores = [scores]
        
    if scores[1] == '未提及':
        scores[1] = score_extract_human(query)
        
    print("scores=",scores)
    
    # 学校必须是一个list,["选科","分数"]
    
    if len(scores) == 2 :
        if scores[0] != "未提及" and scores[1] != "未提及":
            # 分科提取
            try:
                if "文科" in query or "理科" in query or "物理组" in query or "历史组" in query or "综合类" in query:
                    return_format["branch"] = branch2id[scores[0]]
                else:
                    return_format["branch"] = 0
            except Exception as e:
                print("branch validation error is", e )
                return_format["branch"] = 0
            
            # 分数提取
            temp_score = str(scores[1]).replace('分','')
            if '~' in temp_score or '到' in temp_score or '-' in temp_score:
                if '~' in temp_score:
                    low_score = temp_score.split('~')[0]
                    high_score = temp_score.split('~')[1]
                if '到' in temp_score:
                    low_score = temp_score.split('到')[0]
                    high_score = temp_score.split('到')[1]
                if '-' in temp_score:
                    low_score = temp_score.split('-')[0]
                    high_score = temp_score.split('-')[1]
                    
                try:
                    if low_score.isdigit() and high_score.isdigit():
                        return_format["minScore"] = int(low_score)
                        return_format["maxScore"] = int(high_score)
                    else:
                        return_format["minScore"] = 0
                        return_format["maxScore"] = 0
                except Exception as e:
                    print("score validation error is", e )
                    return_format["branch"] = 0
            else:
                try:
                    if temp_score.isdigit():
                        return_format["minScore"] = int(temp_score) - 30
                        return_format["maxScore"] = int(temp_score) + 30 
                    else:
                        return_format["minScore"] = 0
                        return_format["maxScore"] = 0
                except Exception as e:
                    print("score validation error is", e )
                    return_format["branch"] = 0
        
        if scores[0] == "未提及" and scores[1] != "未提及":
            # 分科提取
            return_format["branch"] = 0
            # 分数提取
            temp_score = str(scores[1]).replace('分','')
            try:
                if temp_score.isdigit():
                    return_format["minScore"] = int(temp_score) - 30
                    return_format["maxScore"] = int(temp_score) + 30 
                else:
                    return_format["minScore"] = 0
                    return_format["maxScore"] = 0
            except Exception as e:
                print("score validation error is", e )
                return_format["branch"] = 0
        
                
    if len(scores) == 1 :
        print("scores[0]=",scores[0],"只识别一个")
        # 组/科
        if '组' in scores[0] or '科' in scores[0]:
            return_format["branch"] = branch2id[scores[0]]
            return return_format
        if '分' in scores[0] or str(scores[0]).isdigit :
            temp_score = str(scores[0]).replace('分','')
            if temp_score.isdigit():
                return_format["minScore"] = int(temp_score) - 30
                return_format["maxScore"] = int(temp_score) + 30 
            else:
                return_format["minScore"] = 0
                return_format["maxScore"] = 0
            return return_format
                        
    return return_format


def school_type_validation(school_type) -> list:
    """
    
    检查学校类型是否合法
    
    可能存在的字符串: 985,211,双一流

    Args:
        school_type (_type_): _description_
        
    Returns:
        list: 最终学校类型列表
    """
    
    possible_school_type = ["985","211","双一流"]
    
    final_school_type = []
    
    # 确保school_type是一个list
    if type(school_type) == str:
        school_type = [school_type]
    for t in school_type:
        if t in possible_school_type:
            final_school_type.append(t)
    
    return final_school_type


def school_province_validation(provinces) -> list:
    
    # 确保provinces是一个list
    if type(provinces) != list:
        provinces = [provinces]
    
    final_province = []
    for province in provinces:
        if province == "未提及":
            continue
        res = es.search(index='province', body
                    ={"query": {"match": {"name": province}}})
        if len(res['hits']['hits']) == 0:
            return []
        else:
            temp = {
                'name':res['hits']['hits'][0]['_source']['province_name'],
                'id': res['hits']['hits'][0]['_source']['province_id']
            }
            if temp not in final_province:
                final_province.append(temp)
    return final_province


def locate_province_validation(provinces) -> str:
    
    if type(provinces) != list:
        provinces = [str(provinces)]
    print("provinces=",provinces)
    final_province = []
    for province in provinces:
        if province == "未提及":
            continue
        res = es.search(index='province', body
                    ={"query": {"match": {"name": province}}})
        # print(res)
        if len(res['hits']['hits']) == 0:
            continue
        else:
            temp = {
                'name':res['hits']['hits'][0]['_source']['province_name'],
                'id': res['hits']['hits'][0]['_source']['province_id']
            }
            if temp not in final_province:
                final_province.append(temp)
    if len(final_province) > 0:
        final_province = final_province[0]
    if len(final_province) == 0:
        final_province = None
    return final_province


def year_validation(year) -> int:
    """
    
    年份的合法性检查

    Args:
        year (_type_): _description_

    Returns:
        int: _description_
    """
    
    # 如果year是一个list
    if type(year) == list:
        year = str(year[0])
    
    if year == "未提及":
        return None
    if year.isdigit():
        return int(year)
    else:
        return 2024

def major_validation(majors) -> list:
    """
    
    专业名称的合法性检查

    Args:
        majors (_type_): _description_

    Returns:
        list: _description_
    """
    
    # 确保majors是一个list
    if type(majors) == str:
        majors = [majors]
    
    final_majors = []
    print("majors=",majors)
    for major in majors:
        if major == "未提及":
            continue
        res = es.search(index='major', body={
            "query": {
                "match": {
                    "专业名称": major
                }
            },
            "size": 50,
            "_source": ["专业名称", "专业热度","ID"]
        })
        # print("res=",res)
        if len(res['hits']['hits']) == 0:
            return []
        else:
            # 处理数据
            hits = res['hits']['hits']
            data = [(hit['_source']['专业名称'], hit['_source']['专业热度'],hit['_source']['ID']) for hit in hits]
            
            # 按与查询关键词重叠的字符数降序排列，然后按照热度降序排列
            # data.sort(key=lambda x: (len(set(x[0]) & set(major)), x[1]), reverse=True)
            data.sort(key=lambda x: (lcs_length(x[0], major), x[1]), reverse=True)

            # print("排序结果为",data)
            temp = {
                'name':data[0][0],
                'id': data[0][2]
            }
            if temp not in final_majors:
                final_majors.append(temp)
            
    return final_majors
    
    
def school_validation(schools) -> list:
    """
    
    院校名称的合法性检查

    Args:
        schools (_type_): _description_

    Returns:
        list: _description_
    """
    
    # 确保schools是一个list
    if type(schools) == str:
        schools = [schools]
    
    final_schools = []
    for school in schools:
        if school == "未提及":
            continue
        res = es.search(index='school', body
                        ={"query": {"match": {"院校名称": school}}})
        if len(res['hits']['hits']) == 0:
            return []
        else:
            temp = {
                'name':res['hits']['hits'][0]['_source']['院校名称'],
                'id': res['hits']['hits'][0]['_source']['院校ID']
            }
            if temp not in final_schools:
                final_schools.append(temp)
            # final_schools.append(res['hits']['hits'][0]['_source']['院校名称'])
    return final_schools
    
def format_clean(target_string):
    if "//" in target_string:
        target_string = target_string.split("//")[0]
    if "(" in target_string and ")" in target_string:
        target_string = target_string.split("(")[0]
    return target_string

def answer_type_check(answer):
    answer = answer.replace("：", ":").replace("，",",")
    answer_1 = answer.split('\n')
    answer = ('\n'.join([format_clean(t) for t in answer_1])).replace('\n\n','\n')
    print("answer check/clean answer=",answer)
    try:
        answer = eval(answer)
    except Exception as e:
        print("Exception is ", e)
    return answer

def score_extract_human(query):
    # 600分
    # pattern = r'\d{3}分'  # 匹配3个数字加"分"
    # extract_scores = re.findall(pattern, query)
    # if len(extract_scores) == 0:
    #     # 600
    #     pattern = r'\d{3}'  # 匹配3个数字
    #     extract_scores = re.findall(pattern, query)
    # else:
    #     if len(extract_scores) > 0:
    #         extract_scores = [t.replace('分','') for t in extract_scores]
    # print("最终分数",extract_scores)  # 输出: ['600分', '520分', '680分']
    
    # if len(extract_scores) > 0:
    #     return extract_scores[0]
    # else:
    #     return "未提及"
    
    pattern = r"\d{3}分|\d{3}"
    extract_scores = re.findall(pattern, query)
    if len(extract_scores) == 0:
        return "未提及"
    else:
        extract_scores = [ t.replace('分','') for t in extract_scores ]
    
    return '--'.join(extract_scores)
  

def history_reformat(h) -> {}:
    """防止传入的history有问题，主要是针对UI交互的场景

    Returns:
        _type_: _description_
    """
    res = {"role": h.role, "content": h.content}
    return res
  
  
# bert判断问题是否是闲聊
async def get_idle_res(query):
    url = "http://127.0.0.1:6006/chat/bert_chat_judge"  # 你的目标 URL
    payload = query
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload) as response:
            return await response.text()
        
        
        
def get_next_guidance_info(origin_data):
    """
    
    获取下一步引导信息

    Args:
        origin_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # 如果省份为空
    if origin_data['province'] == None:
        return '请问您在哪个省份？\n 您可以试着说:"我在北京市"'
    
    # 如果年份为空
    if origin_data['year'] == None:
        return '请问您想查哪一年的数据？\n 您可以试着说:"我想查2025年的数据"'
    
    # 如果专业为空
    if len(origin_data['intentionSpecialtyList']) == 0:
        return '请问您有什么想学的专业吗？\n 您可以试着说:"我想学人工智能"'
    
    # 如果分数为空
    if len(origin_data['intentionScoreline']) == 0:
        return '请问您的分数是什么？\n 您可以试着说:"我在福建省物理组考了600分"'
    
    # 如果意向学校所在地为空
    if len(origin_data['intentionProvinces']) == 0:
        return '请问您想差哪个省份大学？\n 您可以试着说:"我想查山东省的大学。"'
    
    # 如果学校类别为空
    if len(origin_data['tags']) == 0:
        return '请问您想去什么类型的学校？\n 您可以试着说:"我想去双一流类型的大学。"'
    
    # 如果意向学校为空
    if len(origin_data['intentionUniversityList']) == 0:
        return '请问您有什么想去的大学吗？\n 您可以试着说:"我想去清华大学，北京大学。"'
    
    # 如果意向科目为空
    if len(origin_data['intentionSubjects']) == 0:
        return '请问您选择了什么科目？\n 您可以试着说:"我选了物化生"'
    
    return "您的信息已经完整，可以进行查询了"

        

async def guidance_info_retrieve(
  query: str = Body(..., examples=["samples"]),
  history: Union[List, List[History]] = Body([],
              description="历史对话，设为一个整数可以从数据库中读取历史消息",
              examples=[[
                  {"role": "user",
                  "content": "我们来玩成语接龙，我先来，生龙活虎"},
                  {"role": "assistant", "content": "虎头虎脑"}]]
              ),
  stream: bool = Body(False, description="流式输出")
  ):
    
    print("query=",query)
    
    origin_data =  {
        "province": None,
        "year": None,
        "intentionSpecialtyList": [],
        "intentionScoreline": [],
        "intentionProvinces": [],
        "tags": [],
        "intentionUniversityList": [],
        "intentionSubjects": []
    }
    
    # 如果有历史记录
    if history:
        # pre_history = history_preprocess(history)
        # history = [{"role": "user", "content":pre_history}] + history 
        origin_data = history_preprocess(history)
        print("已提取纪录",origin_data)

    ret = {
        "status": -1,
        "reason": "",
        "data": origin_data
    }
    
    judge_text = await get_idle_res(query)
    print("bert模型回答", judge_text)
    
    
    async def process_items(prompt_name, semaphore):
        async with semaphore:
            # 在一个单独的线程中运行同步生成器，并获取所有内容
            result = await asyncio.to_thread(lambda: list(api.chat_career(query, history=history, prompt_name=prompt_name)))
            return result
    
    time0 = time.time()
    
    pre_extract_infos = ["career_agent"]
    agent_info = []
    async def pre_process_all_items():
        agent_info = []
        semaphore = asyncio.Semaphore(1)  # 限制并发数为2
        tasks = [process_items(extract_info, semaphore) for extract_info in pre_extract_infos]
        results = await asyncio.gather(*tasks)
        for result, pre_extract_info in zip(results, pre_extract_infos):
            res = ""
            for item in result:
                try:
                    res += item['text']
                except:
                    print('返回有错')
                    ret["data"] = json.dumps(origin_data)
                    return JSONResponse(ret)
            print("agent res=",res)
            try:
                agent_info.extend(eval(str(res)))
            except Exception as e:
                print("agent_info error is", e)
                agent_info = []
        return agent_info

    agent_info = await pre_process_all_items()
    def agent_info_reformat(agent_info):
        res = []
        for item in agent_info:
            print("item=",item)
            extract_info = ''
            if item != "未提及":
                if '考生所在地区' in item:
                    print("识别到地区意图")
                    extract_info = 'career_locate'
                if '年份' in item:
                    print("识别到年份意图")
                    extract_info = 'career_year'
                if '偏好专业' in item:
                    print("识别到专业意图")
                    extract_info = 'career_major'
                if '分数' in item:
                    print("识别到分数意图")
                    extract_info = 'career_school_score'
                if '意向学校所在地' in item:
                    print("识别到意向学校所在地意图")
                    extract_info = 'career_school_province'
                if '学校类别' in item:
                    print("识别到学校类别意图")
                    extract_info = 'career_school_tag'
                if '意向学校' in item:
                    print("识别到意向学校意图")
                    extract_info = 'career_schools'
                if '想查询的科目组合或已选科目' in item:
                    print("识别到科目组合意图")
                    extract_info = 'career_selection'
                if extract_info!='':
                    res.append(extract_info)
                    
        return res
    
    if not '相关' in judge_text and agent_info == []:
        ret['status'] = 401
        ret['reason'] = '意图判断为非选科相关意图'
        ret['data'] = None
        print("非选科相关意图",ret)
        return JSONResponse(ret)

    
    operation_judge = ""
    if agent_info != [] and len(history) > 0:
        extract_infos = agent_info_reformat(agent_info)
        pre_extract_infos = ["career_operation"]
        operation_judge = ""
        async def pre_process_all_items():
            operation_judge = ""
            semaphore = asyncio.Semaphore(1)  # 限制并发数为2
            tasks = [process_items(extract_info, semaphore) for extract_info in pre_extract_infos]
            results = await asyncio.gather(*tasks)
            for result, pre_extract_info in zip(results, pre_extract_infos):
                res = ""
                for item in result:
                    try:
                        res += item['text']
                    except:
                        print('返回有错')
                        ret["data"] = json.dumps(origin_data)
                        return JSONResponse(ret)
                print("agent res=",res)
                try:
                    operation_judge=eval(str(res))
                except Exception as e:
                    print("agent_info error is", e)
                    operation_judge = ""
            return operation_judge
        operation_judge = await pre_process_all_items()
        print("经过agent后,最终意图",extract_infos, "增删改查意图为",operation_judge)
    else:
        if len(history) == 0:
            print("历史为空,使用全意图")
        if agent_info == []:
            print("agent判断失败,最终意图")
        operation_judge = "增加"
        extract_infos = ['career_locate','career_year','career_major','career_school_score','career_school_province','career_school_tag','career_selection','career_schools']

    # extract_infos = ['career_locate','career_year','career_major','career_school_score','career_school_province','career_school_tag','career_selection','career_schools']
 
    all_info = []
    
    time1 = time.time()
    
    
    
    async def process_all_items():
        semaphore = asyncio.Semaphore(3)  # 限制并发数为2
        tasks = [process_items(extract_info, semaphore) for extract_info in extract_infos]
        results = await asyncio.gather(*tasks)
        for result, extract_info in zip(results, extract_infos):
            res = ""
            for item in result:
                try:
                    res += item['text']
                except:
                    print('返回有错')
                    ret["data"] = json.dumps(origin_data)
                    return JSONResponse(ret)
            
            res = res.replace("：", ":").replace("，", ",")
            # print("用时", time.time() - time0, "秒")
            all_info.append([res, extract_info])

    await process_all_items()
        
    wrong_flags = []
    for t in zip(all_info,extract_infos):
        print(t[1],t[0])
    for info in all_info:
        origin_data,wrong_flag = extract_info_sep(info,origin_data,query, operation_judge)
        print(info,"正确还是错",wrong_flag)
        wrong_flags.append(wrong_flag)
  
        
    # print("最终发送数据=",origin_data)
    
    # 获取引导信息
    
    content = get_next_guidance_info(origin_data)
    
    final_time = time.time()
    print("总用时",final_time-time0,"秒", "agent用时",time1-time0,"秒", "extract用时",final_time-time1,"秒")
    ret['content'] = content
    ret['data'] = json.dumps(origin_data)


    if True in wrong_flags:
        ret['status'] = 501
        ret['reason'] = '数据提取失败'
    else:
        ret['status'] = 200
        ret['reason'] = 'success'
    ret['time_spent'] = {
        "总用时": final_time-time0,
        "agent用时": time1-time0,
        "extract用时": final_time-time1
    }
    print("最终结果",ret)
    
    return JSONResponse(ret)
