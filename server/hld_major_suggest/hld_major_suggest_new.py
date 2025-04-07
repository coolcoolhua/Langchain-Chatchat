from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
from fastapi import Body, Request


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd

overlap= json.load(open('./server/hld_major_suggest/清洗后数据/overlap.json'))
# 职业和专业的对应关系，要找badcase都从这里改
majors = json.load(open('./server/hld_major_suggest/清洗后数据/code2major_v1.json'))
major2largemajor = json.load(open('./server/hld_major_suggest/清洗后数据/major2largemajor.json'))
extra_data = json.load(open('./server/hld_major_suggest/清洗后数据/hld2专业类按热度降序.json'))

# 整理职业的interest为向量
job_data = json.load(open('./server/hld_major_suggest/清洗后数据/code2interests.json'))
# # print(job_data)
scores = []
names = []
codes = []
for key,item in job_data.items():
    name =  item["name"]
    # if 'Teachers' in name:
    #     continue
    names.append(name)
    codes.append(key)
    score = [t[1] for t in item["score"].items()]
    scores.append(score)
    

# 职业数据
vecs = np.array(scores)
vecs_normalized = (vecs - 0) / (1 - 0)

# 计算每个向量的均值和标准差
vecs_means = np.mean(vecs_normalized, axis=1)
vecs_stds = np.std(vecs_normalized, axis=1)


def get_majors(code):
    return majors.get(code, [])


def rerank_good_majors(majors,max_dimension):
    # 匹配度从高到低召回专业
    # 根据专业的热度重新排序majors
    good_major = []
    good_major_detail = []
    for index,major in enumerate(majors):
        # if index<20:
        #     print("推荐专业",index,major)
        #     pass
        if major[0][-1] < 0.3:
            continue
        for m in major[1]:
            if m not in good_major and major[-1][0] == max_dimension:
                # print("推荐专业",m,"专业最高码",major[-1][0],"最高维度",max_dimension)
                good_major.append(m)
                good_major_detail.append(
                    {
                        "专业":m,
                        "专业最高码":major[-1][0],
                        "最高维度":max_dimension,
                        "推荐度":major[0][-1],
                        "具体内容": major[1]
                    }
                )
    # print("重排后的推荐专业")
    # for index,t in enumerate(good_major_detail[:20]):
    #     print(index,t)
    return good_major, good_major_detail

def rerank_bad_majors(majors):
    # 先逆序排匹配度，逆序从高到低(-1到0)召回专业
    bad_major = []
    for index,major in enumerate(majors):
            # print("不推荐专业",index,major)
        # 一些特殊的分数型可能会导致不召回任何专业
        if major[0][-1] > 0:
            # print("分数过高",major)
            continue
        for m in major[1]:
            if m not in bad_major:
               bad_major.append([m,major[0][-1]])
    print("最终不推荐专业的数量",len(bad_major))
    return bad_major

def get_top3(vec):
    # 获取该向量从大到小前三位的index
    top3_indices = np.argsort(vec)[::-1][:3]
    return top3_indices
  
def get_interest_code(code):
    if code == 0:
        return 'A'
    if code == 1:
        return 'E'
    if code == 2:
        return 'S'
    if code == 3:
        return 'R'
    if code == 4:
        return 'C'
    if code == 5:
        return 'I'
      
      
def rank_stategy(job,job_codes, student_codes):
    """
    首先匹配三码的，三码中都被推荐的专业，就按照关联次数来排序
    然后匹配双码且顺序相同的，继续验证关联次数排序
    然后再匹配双码顺序不同的

    Args:
        codes (_type_): _description_
    """
    
    return job
  
  
def lowrank_stategy(job,job_codes, student_codes):
    """
    首先匹配三码的，三码中都被推荐的专业，就按照关联次数来排序
    然后匹配双码且顺序相同的，继续验证关联次数排序
    然后再匹配双码顺序不同的

    Args:
        codes (_type_): _description_
    """

    return job
  
  
def expand_major_large(major_list,interest_code):
    # # print("当前专业类",major_list)
    if len(major_list)>15:
        # print("最终返回",major_list[:15])
        return major_list[:15]
    # 找前两个
    extra = extra_data.get(interest_code[:2], [])
    # # print("前两个",extra)
    for e in extra:
        if e not in major_list:
            major_list.append(e)
    # # print("前两位代码后，当前有",len(major_list),"个专业类")
    # # print("当前专业类",major_list)
    if len(major_list)>15:
        # # print("最终返回",major_list[:15])
        return major_list[:15]
    else:
        extra = extra_data.get(interest_code[:2][::-1], [])
        # # print("前两位逆序",extra)
        for e in extra:
            if e not in major_list:
                major_list.append(e)
        # # print("前两位代码逆序后，当前有",len(major_list),"个专业类")
        if len(major_list)>15:
            # # print("最终返回",major_list[:15])
            return major_list[:15]
        else:
            extra = extra_data.get(interest_code[0], [])
            # # print("第一个",extra)
            for e in extra:
                if e not in major_list:
                    major_list.append(e)
            # # print("只找第一位代码，当前有",len(major_list),"个专业类")
            if len(major_list)>15:
                # # print("最终返回",major_list[:15])
                return major_list[:15]
              
              
              
import time

def get_similar_majors(vec,interest_code,max_dimension):
    总开始时间 = time.time()

    # 假设vec是您的原始向量，vecs是包含1000个向量的数组
    vec_old = vec
    vec = np.array(vec)
    # 归一化vec,vecs
    # 将vec和vecs归一化到[0, 1]范围
    # vec_normalized = (vec - 13.333) / (66.667 - 13.333)
    vec_normalized = (vec - 0) / (100 - 0)
    # 保留两位小数
    vec_normalized = np.round(vec_normalized, 2)
    
    # 计算vec的均值
    vec_mean = np.mean(vec_normalized)
    # 计算vec的标准差
    vec_std = np.std(vec_normalized)

    向量处理开始时间 = time.time()
    numerators = np.sum((vec_normalized - vec_mean) * (vecs - vecs_means[:, np.newaxis]), axis=1)
    
    N = 6
    denominators = N * vec_std * vecs_stds
    
    correlations = numerators / denominators
    
    similarities = correlations
    向量处理结束时间 = time.time()
    # print(f"向量处理时间: {向量处理结束时间 - 向量处理开始时间:.4f} 秒")

    # 打印相似度结果
    top_20_indices = np.argsort(similarities)[::-1]
    
    推荐职业开始时间 = time.time()
    recommend_jobs = []
    recommend_majors = []
    
    for n, idx in enumerate(top_20_indices):
        recommend_jobs.append([names[idx], vecs_normalized[idx],similarities[idx]])
        recommend_majors.append(get_majors(codes[idx]))
    
    # 推荐的统一为一个list
    rec_jobs = []
    # 不推荐的也统一为一个list
    unrec_jobs = []
    
    for t in zip(recommend_jobs, recommend_majors):
        indexes = get_top3(t[0][1])
        job_codes = ''.join([get_interest_code(t) for t in indexes])
        t = t + (job_codes,)
            
        res = rank_stategy(t,job_codes,interest_code)
        rec_jobs.append(res)
        bad_res = lowrank_stategy(t,job_codes,interest_code)
        unrec_jobs.append(bad_res)
    print("推荐职业前5")
    for index,r in enumerate(rec_jobs[:5]):
        print(index,r)
    unrec_jobs = unrec_jobs[::-1]
    print("不推荐职业前5")
    for index,u in enumerate(unrec_jobs[:5]):
        print(index,u)
    推荐职业结束时间 = time.time()
    # print(f"推荐职业处理时间: {推荐职业结束时间 - 推荐职业开始时间:.4f} 秒")
            
    
    专业处理开始时间 = time.time()
    
    # 推荐专业处理
    推荐专业开始时间 = time.time()
    all_rec_jobs = []
    good_major_rerank_res, good_major_rerank_detail = rerank_good_majors(rec_jobs,max_dimension)
    for t in good_major_rerank_res:
        if t not in all_rec_jobs:
            all_rec_jobs.append(t)
    推荐专业结束时间 = time.time()
    # print(f"推荐专业处理时间: {推荐专业结束时间 - 推荐专业开始时间:.4f} 秒")
    
    # 推荐专业大类处理
    推荐专业大类开始时间 = time.time()
    rec_jobs_large = []
    for t in [major2largemajor.get(t, '') for t in all_rec_jobs if major2largemajor.get(t, '') != '']:
        if t not in rec_jobs_large:
            rec_jobs_large.append(t)
    推荐专业大类结束时间 = time.time()
    # print(f"推荐专业大类处理时间: {推荐专业大类结束时间 - 推荐专业大类开始时间:.4f} 秒")
    print("所有推荐专业大类",rec_jobs_large)
    
    # 不推荐专业处理
    不推荐专业开始时间 = time.time()
    all_unrec_jobs = []
    unrec_score_dict = {}
    for t in rerank_bad_majors(unrec_jobs):
        if t[0] not in all_unrec_jobs:
            all_unrec_jobs.append(t[0])
            unrec_score_dict[t[0]] = t[1]
    # 根据分数进行排序
    unrec_score_dict = dict(sorted(unrec_score_dict.items(), key=lambda item: item[1]))
    不推荐专业结束时间 = time.time()
    # print(f"不推荐专业处理时间: {不推荐专业结束时间 - 不推荐专业开始时间:.4f} 秒")
    
    # 不推荐专业大类处理
    不推荐专业大类开始时间 = time.time()
    unrec_jobs_large = []
    for t in [major2largemajor.get(t, '') for t in all_unrec_jobs if major2largemajor.get(t, '') != '']:
        if t not in unrec_jobs_large:
            unrec_jobs_large.append(t)
    不推荐专业大类结束时间 = time.time()
    # print(f"不推荐专业大类处理时间: {不推荐专业大类结束时间 - 不推荐专业大类开始时间:.4f} 秒")
    print("所有不推荐专业大类",unrec_jobs_large)
    
    # 专业大类重叠处理
    专业大类重叠开始时间 = time.time()
    # all_rec_check = []
    # for t in all_rec_jobs:
    #     if t not in all_rec_check:
    #         all_rec_check.append([ [t,major2largemajor.get(t, '')] for t in all_rec_jobs if major2largemajor.get(t, '') != ''])
    
    all_rec_large = []
    for t in [major2largemajor.get(t, '') for t in all_rec_jobs if major2largemajor.get(t, '') != '']:
        if t not in all_rec_large:
            all_rec_large.append(t)
    专业大类重叠结束时间 = time.time()
    print(f"专业大类重叠处理时间: {专业大类重叠结束时间 - 专业大类重叠开始时间:.4f} 秒")
    
    # 专业大类扩展处理
    专业大类扩展开始时间 = time.time()
    expand_flag = False
    if len(all_rec_large) < 15:
        expand_flag = True
        all_rec_large = expand_major_large(all_rec_large,interest_code)
    
    all_rec_large = all_rec_large[:15]
    专业大类扩展结束时间 = time.time()
    # print(f"专业大类扩展处理时间: {专业大类扩展结束时间 - 专业大类扩展开始时间:.4f} 秒")
    
    # 不推荐专业检查处理
    # 不推荐专业检查开始时间 = time.time()
    # all_unrec_check  = []
    # for t in all_unrec_jobs:
    #     if t not in all_unrec_check:
    #         all_unrec_check.append([[t,major2largemajor.get(t, '')] for t in all_unrec_jobs if major2largemajor.get(t, '') != ''])
    # 不推荐专业检查结束时间 = time.time()
    # print(f"不推荐专业检查处理时间: {不推荐专业检查结束时间 - 不推荐专业检查开始时间:.4f} 秒")
    
    # 不推荐大类排序处理
    不推荐大类排序开始时间 = time.time()
    all_unrec_large = []
    temp = {}
    for t in all_unrec_jobs:
        large_major = major2largemajor.get(t, '')
        if large_major not in all_rec_large:
            if large_major not in temp:
                temp[large_major] = [unrec_score_dict.get(t, 0)]
            else:
                temp[large_major].append(unrec_score_dict.get(t, 0))
    temp = dict(sorted(temp.items(), key=lambda item: np.mean(item[1])))
    for t in temp:
        all_unrec_large.append(t)
    不推荐大类排序结束时间 = time.time()
    # print(f"不推荐大类排序处理时间: {不推荐大类排序结束时间 - 不推荐大类排序开始时间:.4f} 秒")
    
    专业处理结束时间 = time.time()
    # print(f"专业处理总时间: {专业处理结束时间 - 专业处理开始时间:.4f} 秒")

    总结束时间 = time.time()
    总执行时间 = 总结束时间 - 总开始时间
    print(f"总执行时间: {总执行时间:.4f} 秒")

    res = {
        "学生分数":f"A:{vec_old[0]}\nE:{vec_old[1]}\nS:{vec_old[2]}\nR:{vec_old[3]}\nC:{vec_old[4]}\nI:{vec_old[5]}",
        "兴趣代码排序": interest_code,
        "所有匹配专业(最多前15)": all_rec_jobs[:15],
        "所有匹配专业大类(最多前15)": (all_rec_large[:15]),
        "是否进行了专业大类扩展": expand_flag,
        "推荐职业明细": rec_jobs[:100],
        "所有不匹配专业(最多前15)": all_unrec_jobs[:15],
        "所有不匹配专业大类(最多前15)": all_unrec_large[:15],
        "不推荐职业明细": unrec_jobs[:100],
        # "冲突专业": ','.join(overlap_check),
        # "冲突专业对应的专业大类": ','.join(overlap_check_large),
        "高亮专业": all_rec_jobs,
        "重排后的推荐专业": good_major_rerank_detail,
        "执行时间": {
            "总执行时间": f"{总执行时间:.4f} 秒",
            "向量处理时间": f"{向量处理结束时间 - 向量处理开始时间:.4f} 秒",
            "推荐职业处理时间": f"{推荐职业结束时间 - 推荐职业开始时间:.4f} 秒",
            "专业处理时间": f"{专业处理结束时间 - 专业处理开始时间:.4f} 秒"
        }
    }
    
    return res


def hld_major_suggest_new(
  student_score: dict = Body(..., examples=[{"A":43,
        "E":38,
        "S":43,
        "R":50,
        "C":38,
        "I":45}]),
  interest_code: str = Body(..., examples=["AESRCI"])
  ):
    # print(student_score,interest_code)
  
    a_score = student_score.get("A", 0)
    e_score = student_score.get("E", 0)
    s_score = student_score.get("S", 0)
    r_score = student_score.get("R", 0)
    c_score = student_score.get("C", 0)
    i_score = student_score.get("I", 0)
    
    student_score = [a_score,e_score,s_score,r_score,c_score,i_score]
    
    # 计算6个维度分数最高的维度
    max_score = max(student_score)
    max_dimension = "AESRCI"[student_score.index(max_score)]
    print(f"最高分数的维度是: {max_dimension}, 分数为: {max_score}")
  
    res = get_similar_majors(student_score,interest_code,max_dimension)
    good_jobs = [{"职业名":t[0][0],"霍兰德码":t[2],"推荐度":str(t[0][-1]),"推荐专业":t[1]} for t in res["推荐职业明细"]]
    bad_jobs = [{"职业名":t[0][0],"霍兰德码":t[2],"推荐度":str(t[0][-1]),"不推荐专业":t[1]} for t in res["不推荐职业明细"]]
    # # print("here",bad_jobs)
    return_res = {
        "good_majors": res["所有匹配专业大类(最多前15)"],
        "bad_majors":  res["所有不匹配专业大类(最多前15)"],
        "key_majors": res["高亮专业"],
        "good_jobs": good_jobs,
        "bad_jobs": bad_jobs,
        "extend": res["是否进行了专业大类扩展"],
        "rerank_major_list": json.dumps(res["重排后的推荐专业"])
    }
    # print("推荐",return_res["good_majors"])
    # print("不推荐",return_res["bad_majors"])
    
    
    ret = return_res
    print(ret)
    
    return JSONResponse(ret)
