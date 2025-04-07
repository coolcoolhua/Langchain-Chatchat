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
majors = json.load(open('./server/hld_major_suggest/清洗后数据/code2major.json'))
major2largemajor = json.load(open('./server/hld_major_suggest/清洗后数据/major2largemajor.json'))
extra_data = json.load(open('./server/hld_major_suggest/清洗后数据/hld2专业类按热度降序.json'))

# 整理职业的interest为向量
job_data = json.load(open('./server/hld_major_suggest/清洗后数据/code2interests.json'))
# print(job_data)
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
vecs_normalized = (vecs - 0) / (100 - 0)


def get_majors(code):
    return majors.get(code, [])

def rerank_majors(majors):
    # majors是一个list,里面每个元素也是list,代表一个职业对应的专业,比如majors[0] = ['计算机科学与技术', '软件工程']
    # 根据专业的热度重新排序majors
    major2count = {}
    for major in majors:
        # print(major)
        for m in major[1]:
            if m not in major2count:
                major2count[m] = 1
            else:
                major2count[m] += 1
    # print(major2count)
    sorted_major2count = sorted(major2count.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_major2count)
    sorted_majors = [t[0] for t in sorted_major2count]
    return sorted_majors

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
    
    if job_codes[:3] == student_codes[:3]:
        # print("三码匹配")     
        # print('推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
            
        return [3,job]
    
            
    if job_codes[:2] == student_codes[:2]:
        # print("双码匹配")     
        # print('推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
        return [2,job]
    
    if job_codes[:2] == (student_codes[:2])[::-1]:
        # print("双码逆序匹配")     
        # print('推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
        return [1,job]
    
    return [-1,job]
  
  
def lowrank_stategy(job,job_codes, student_codes):
    """
    首先匹配三码的，三码中都被推荐的专业，就按照关联次数来排序
    然后匹配双码且顺序相同的，继续验证关联次数排序
    然后再匹配双码顺序不同的

    Args:
        codes (_type_): _description_
    """

    if job_codes[:3] == student_codes[::-1][:3]:
        # print("三码不匹配")     
        # print('不推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
            
        return [3,job]

            
    if job_codes[:2] == student_codes[::-1][:2]:
        # print("双码不匹配")     
        # print('不推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
        return [2,job]

    if job_codes[:2] == (student_codes[::-1][:2][::-1]):
        # print("双码逆序不匹配")     
        # print('不推荐职业',job[0][0],'优势兴趣代码',job_codes)
        # print('对应向量',job[0][1],'相似度',job[0][2])
        if len(job[1])<1:
            # print(job[0][0],"职业未关联专业")
            possible_jobs = overlap.get(job[0][0], [])
            # print("可能关联专业", possible_jobs)
        else:
            # print('关联专业',job[1])
            pass
        return [1,job]

    return [-1,job]
  
  
def expand_major_large(major_list,interest_code):
    # print("当前专业类",major_list)
    if len(major_list)>15:
        print("最终返回",major_list[:15])
        return major_list[:15]
    # 找前两个
    extra = extra_data.get(interest_code[:2], [])
    # print("前两个",extra)
    for e in extra:
        if e not in major_list:
            major_list.append(e)
    # print("前两位代码后，当前有",len(major_list),"个专业类")
    # print("当前专业类",major_list)
    if len(major_list)>15:
        # print("最终返回",major_list[:15])
        return major_list[:15]
    else:
        extra = extra_data.get(interest_code[:2][::-1], [])
        # print("前两位逆序",extra)
        for e in extra:
            if e not in major_list:
                major_list.append(e)
        # print("前两位代码逆序后，当前有",len(major_list),"个专业类")
        if len(major_list)>15:
            # print("最终返回",major_list[:15])
            return major_list[:15]
        else:
            extra = extra_data.get(interest_code[0], [])
            # print("第一个",extra)
            for e in extra:
                if e not in major_list:
                    major_list.append(e)
            # print("只找第一位代码，当前有",len(major_list),"个专业类")
            if len(major_list)>15:
                # print("最终返回",major_list[:15])
                return major_list[:15]
              
              
              
def get_similar_majors(vec,interest_code):
    # 假设vec是您的原始向量，vecs是包含1000个向量的数组
    vec_old = vec
    vec = np.array(vec)
    # 归一化vec,vecs
    # 将vec和vecs归一化到[0, 1]范围
    # vec_normalized = (vec - 13.333) / (66.667 - 13.333)
    vec_normalized = (vec - 0) / (100 - 0)
    # 保留两位小数
    vec_normalized = np.round(vec_normalized, 2)
    
    # 计算余弦相似度
    cosine_sims = np.dot(vec_normalized, vecs_normalized.T)
    
    # 计算欧几里得距离
    euclidean_dists = np.linalg.norm(vec_normalized - vecs_normalized, axis=1)
    
    # 计算综合相似度得分
    similarities = cosine_sims * (1 / (1 + euclidean_dists))

    # 打印相似度结果
    top_20_indices = np.argsort(similarities)[::-1]
    # 打印前20个最相似向量的索引和相似度值
    # 艺术型（A）、企业型（E）、社会型（S）、实际型（R）、传统型（C）、研究型（I）
    print("艺术型（A）、企业型（E）、社会型（S）、实际型（R）、传统型（C）、研究型（I）")
    print(f"学生分数\nA:{vec_old[0]}\nE:{vec_old[1]}\nS:{vec_old[2]}\nR:{vec_old[3]}\nC:{vec_old[4]}\nI:{vec_old[5]}")
    print("优势兴趣代码", interest_code)
    print("归一化后分数", vec_normalized)
    
    recommend_jobs = []
    recommend_majors = []
    print("前5个向量")
    print(similarities[top_20_indices[:5]])
    
    for n, idx in enumerate(top_20_indices):
        recommend_jobs.append([names[idx], vecs_normalized[idx],similarities[idx]])
        recommend_majors.append(get_majors(codes[idx]))
        # print(f"Similarity {i+1}: Index={idx}, Similarity={similarities[idx]}, Name = {names[idx]} , Codes = {codes[idx]}, Majors = {get_majors(codes[idx])}")
    
    rec_3_jobs = []
    rec_2_jobs = []
    rec_1_jobs = []
    unrec_3_jobs = []
    unrec_2_jobs = []
    unrec_1_jobs = []
    
    # print("重新排序后的专业", rerank_majors(recommend_majors))
    for t in zip(recommend_jobs, recommend_majors):
        # for job in jobs[
        # if len(t[1])==0:
        #     continue
        indexes = get_top3(t[0][1])
        job_codes = ''.join([get_interest_code(t) for t in indexes])
        t = t + (job_codes,)
        res = rank_stategy(t,job_codes,interest_code)
        if res[0] == 3:
            rec_3_jobs.append(res[1])
        if res[0] == 2:
            rec_2_jobs.append(res[1])
        if res[0] == 1:
            rec_1_jobs.append(res[1])
            
        res1 = lowrank_stategy(t,job_codes,interest_code)
        if res1[0] == 3:
            unrec_3_jobs.append(res1[1])
        if res1[0] == 2:
            unrec_2_jobs.append(res1[1])
        if res1[0] == 1:
            unrec_1_jobs.append(res1[1])
            
    conflict_check_rec= rec_3_jobs + rec_2_jobs + rec_1_jobs
    conflict_check_unrec= unrec_3_jobs + unrec_2_jobs + unrec_1_jobs
    conflict_res = []
    conflict_job = []
    conflict_unjob = [] 
    for unjob in conflict_check_unrec:
        for job in conflict_check_rec:
            for m in unjob[1]:
                if m in job[1]:
                    conflict_res.append(m)
                    
                    conflict_unjob.append(' '.join(['不推荐',str(unjob[0][0]),"对应代码是",unjob[2],str(unjob[1])])+'\n')
                    conflict_job.append(' '.join(['推荐',str(job[0][0]),"对应代码是",job[2],str(job[1])])+'\n')
                    
                    
    # print("----三码匹配",rec_3_jobs)
    rec_3 = []
    for t in rerank_majors(rec_3_jobs):
        if t not in rec_3:
            rec_3.append(t)
    rec_3_large = []
    for t in [major2largemajor.get(t, '') for t in rec_3 if major2largemajor.get(t, '') != '']:
        if t not in rec_3_large:
            rec_3_large.append(t)
    rec_2 = []
    for t in rerank_majors(rec_2_jobs):
        if t not in rec_2:
            rec_2.append(t)
    rec_2_large = []
    for t in [major2largemajor.get(t, '') for t in rec_2 if major2largemajor.get(t, '') != '']:
        if t not in rec_2_large:
            rec_2_large.append(t)
    rec_1 = []
    for t in rerank_majors(rec_1_jobs):
        if t not in rec_1:
            rec_1.append(t)
    rec_1_large = []
    for t in [major2largemajor.get(t, '') for t in rec_1 if major2largemajor.get(t, '') != '']:
        if t not in rec_1_large:
            rec_1_large.append(t)
            
    
    # print("----三码不匹配",unrec_3_jobs)
    unrec_3 = []
    for t in rerank_majors(unrec_3_jobs):
        if t not in unrec_3:
            unrec_3.append(t)
    unrec_3_large = []
    for t in [major2largemajor.get(t, '') for t in unrec_3 if major2largemajor.get(t, '') != '']:
        if t not in unrec_3_large:
            unrec_3_large.append(t)
    unrec_2 = []
    for t in rerank_majors(unrec_2_jobs):
        if t not in unrec_2:
            unrec_2.append(t)
    unrec_2_large = []
    for t in [major2largemajor.get(t, '') for t in unrec_2 if major2largemajor.get(t, '') != '']:
        if t not in unrec_2_large:
            unrec_2_large.append(t)
    unrec_1 = []
    for t in rerank_majors(unrec_1_jobs):
        if t not in unrec_1:
            unrec_1.append(t)
    unrec_1_large = []
    for t in [major2largemajor.get(t, '') for t in unrec_1 if major2largemajor.get(t, '') != '']:
        if t not in unrec_1_large:
            unrec_1_large.append(t)
            
            
            
    all_rec = []
    for t in rec_3 + rec_2 + rec_1:
        if t not in all_rec:
            all_rec.append(t)
    print("所有匹配专业",all_rec)
    
    # 确定专业大类重叠情况
    all_rec_check = []
    for t in rec_3 + rec_2 + rec_1:
        if t not in all_rec_check:
            all_rec_check.append([ [t,major2largemajor.get(t, '')] for t in all_rec if major2largemajor.get(t, '') != ''])
    
    # print("对应情况",all_rec_check)
    
    all_rec_large = []
    for t in [major2largemajor.get(t, '') for t in all_rec if major2largemajor.get(t, '') != '']:
        if t not in all_rec_large:
            all_rec_large.append(t)
    print("所有匹配专业大类",all_rec_large)
    all_rec_large = expand_major_large(all_rec_large,interest_code)
    print("扩展后，所有匹配专业大类",all_rec_large)
    
    all_unrec = []
    for t in unrec_3 + unrec_2 + unrec_1:
        if t not in all_unrec:
            all_unrec.append(t)
    print("所有不匹配专业",all_unrec)
    
    all_unrec_check  = []
    for t in unrec_3 + unrec_2 + unrec_1:
        if t not in all_unrec_check:
            all_unrec_check.append([[t,major2largemajor.get(t, '')] for t in all_unrec if major2largemajor.get(t, '') != ''])
    # print("对应情况",all_unrec_check)
    
    all_unrec_large = []
    for t in [major2largemajor.get(t, '') for t in all_unrec if major2largemajor.get(t, '') != '']:
        if t not in all_unrec_large:
            # 增加解决冲突逻辑
            if t not in all_rec_large:
                all_unrec_large.append(t)
    print("所有不匹配专业大类",all_unrec_large)
    
    
    res = {
        "学生分数":f"A:{vec_old[0]} E:{vec_old[1]} S:{vec_old[2]} R:{vec_old[3]} C:{vec_old[4]} I:{vec_old[5]}",
        "兴趣代码排序": interest_code,
        "所有匹配专业(最多前15)": ','.join(all_rec[:15]),
        "所有匹配专业大类(最多前15)": all_rec_large[:15],
        "三码匹配专业": ','.join(rec_3),
        "三码匹配专业大类": ','.join(rec_3_large),
        "双码匹配专业": ','.join(rec_2),
        "双码匹配专业大类": ','.join(rec_2_large),
        "双码逆序匹配专业": ','.join(rec_1),
        "双码逆序匹配专业大类": ','.join(rec_1_large),
        "所有不匹配专业(最多前15)": ','.join(all_unrec[:5]),
        "所有不匹配专业大类(最多前15)": all_unrec_large[:5],
        "三码不匹配专业": ','.join(unrec_3),
        "三码不匹配专业大类": ','.join(unrec_3_large),
        "双码不匹配专业": ','.join(unrec_2),
        "双码不匹配专业大类": ','.join(unrec_2_large),
        "双码逆序不匹配专业": ','.join(unrec_1),
        "双码逆序不匹配专业大类": ','.join(unrec_1_large),
        "冲突专业": ','.join(list(set(conflict_res))),
        "不推荐职业及其关联专业": ''.join(list(set(conflict_unjob))),
        "推荐职业及其关联专业": ''.join(list(set(conflict_job)))
    }
    
    details3 = ""
    for job in rec_3_jobs:
        details3 += f"推荐职业 {job[0][0]} 优势兴趣代码 {job[2]} 对应向量 {job[0][1]} 相似度 {job[0][2]} 关联专业 {job[1]}\n"
    details2 = ""
    for job in rec_2_jobs:
        details2 += f"推荐职业 {job[0][0]} 优势兴趣代码 {job[2]} 对应向量 {job[0][1]} 相似度 {job[0][2]} 关联专业 {job[1]}\n"
    details1 = ""
    for job in rec_1_jobs:
        details1 += f"推荐职业 {job[0][0]} 优势兴趣代码 {job[2]} 对应向量 {job[0][1]} 相似度 {job[0][2]} 关联专业 {job[1]}\n"
        
    res["三码匹配详情"] = details3
    res["双码匹配详情"] = details2
    res["双码逆序匹配详情"] = details1
    
    return res
    


def hld_major_suggest(
  student_score: dict = Body(..., examples=[{"A":43,
        "E":38,
        "S":43,
        "R":50,
        "C":38,
        "I":45}]),
  interest_code: str = Body(..., examples=["AESRCI"])
  ):
    print(student_score,interest_code)
  
    a_score = student_score.get("A", 0)
    e_score = student_score.get("E", 0)
    s_score = student_score.get("S", 0)
    r_score = student_score.get("R", 0)
    c_score = student_score.get("C", 0)
    i_score = student_score.get("I", 0)
    
    student_score = [a_score,e_score,s_score,r_score,c_score,i_score]
  
    res = get_similar_majors(student_score,interest_code)
    
    return_res = {
        "good_majors": res["所有匹配专业大类(最多前15)"],
        "bad_majors":  res["所有不匹配专业大类(最多前15)"],
        "key_majors": res["所有匹配专业(最多前15)"]
    }
    
    ret = return_res
    
    return JSONResponse(ret)
