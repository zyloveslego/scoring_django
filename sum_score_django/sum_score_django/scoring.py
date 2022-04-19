from django.shortcuts import render
from django.views.decorators import csrf
from django.conf import settings

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')

# import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import os
# import matplotlib.pyplot as plt
# import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import itertools
from nltk.tokenize import sent_tokenize

model = settings.MY_MODEL

# 输入摘要 原文 返回分数
# def comb_scoring(input_summary, ori_doc):
#     # input_summary = ""
#     # ori_doc = ""
#
#     # ori_doc_sents = []
#
#     # # 取原文
#     # file = open("/Users/zhouyou/Documents/git/test/summary_scoring_test/ori_doc.txt")
#     #
#     # for i in file.readlines():
#     #     ori_doc += i.replace("\n", "") + " "
#     #     ori_doc_sents.append(i)
#     #
#     # file.close()
#     #
#     # # 取摘要
#     # file = open("/Users/zhouyou/Documents/git/test/summary_scoring_test/input_summary.txt")
#     #
#     # for i in file.readlines():
#     #     input_summary += i.replace("\n", "") + " "
#     #
#     # file.close()
#
#     ori_doc_sents = sent_tokenize(ori_doc)
#
#     ori_doc_embedding = model.encode(ori_doc).reshape(1, -1)
#
#     input_summary_embedding = model.encode(input_summary).reshape(1, -1)
#
#     input_summary_len = len(word_tokenize(input_summary))
#
#     ori_single_max_summ_cos = 0  # g*
#
#     for i in range(3, 10):
#         print(i)
#         flag = 0
#         for sents in itertools.combinations(ori_doc_sents, i):
#             word_limit = int(input_summary_len) + 5
#             word_count = 0
#             temp_single_summ = ""
#             # print(sents)
#             # print("--------------")
#             for ele in sents:
#                 # print(ele)
#                 text_tokens = word_tokenize(ele)
#                 # print(b)
#                 tokens_without_sw = [word for word in text_tokens if word not in string.punctuation]
#                 # print(tokens_without_sw)
#                 # print(len(tokens_without_sw))
#
#                 word_count = word_count + len(tokens_without_sw)
#                 temp_single_summ = temp_single_summ + " " + ele
#
#
#             if word_count <= word_limit:
#                 flag = 1
#
#             if int(input_summary_len * 0.8) + 2 < word_count <= int(input_summary_len) + 2:
#                 # print(temp_single_summ)
#
#                 temp_single_summ_embedding = model.encode(temp_single_summ).reshape(1, -1)
#                 ori_temp_single_summ_cos = cosine_similarity(ori_doc_embedding, temp_single_summ_embedding)[0][0]
#                 # ori_single_summ_cos.append(ori_temp_single_summ_cos)
#                 # print(ori_temp_single_summ_cos)
#
#                 if ori_temp_single_summ_cos > ori_single_max_summ_cos:
#                     ori_single_max_summ_cos = ori_temp_single_summ_cos
#                     # print(temp_single_summ)
#                     # print(ori_single_max_summ_cos)
#
#             # break
#
#         if flag == 0:
#             print("over")
#             break
#
#         # break
#
#     input_summary_cos = cosine_similarity(ori_doc_embedding, input_summary_embedding)[0][0]
#
#     # print(input_summary)
#     print("输入的相似度:")
#     print(input_summary_cos)
#     print("最好的相似度:")
#     print(ori_single_max_summ_cos)
#
#     final_score = int(input_summary_cos * 100 / ori_single_max_summ_cos)
#     if final_score < 0:
#         final_score = 0
#
#     print(final_score)
#
#     return final_score


# 近似摘要分数
def scoring(input_summary, ori_doc, sum_length):
    ori_doc_sents = sent_tokenize(ori_doc)

    ori_doc_word_count = len(word_tokenize(ori_doc))

    ori_doc_embedding = model.encode(ori_doc).reshape(1, -1)

    input_summary_embedding = model.encode(input_summary).reshape(1, -1)

    input_summary_len = len(word_tokenize(input_summary))

    input_summary_cos = cosine_similarity(ori_doc_embedding, input_summary_embedding)[0][0]

    ori_single_max_summ_cos = 0  # g*
    
    sents_cos = {}
    for sent in ori_doc_sents:
        sent_embedding = model.encode(sent).reshape(1, -1)
        sent_cos = cosine_similarity(ori_doc_embedding, sent_embedding)[0][0]
        sents_cos.setdefault(sent, sent_cos)

    # 单句排序
    sents_cos = sorted(sents_cos.items(), key=lambda item: item[1], reverse=True)
    # print(sents_cos)

    sent_limit = 10 if 10 > len(ori_doc_sents) else len(ori_doc_sents)

    if sum_length == "":
        word_limit = int(ori_doc_word_count * 0.3 * 1.1)

    else:
        word_limit = int(sum_length)
        # print(word_limit)
        # print(type(word_limit))

    max_summ = ""

    temp_max_summ, _ = sents_cos[0]
    temp_max_summ_sim = 0

    while(True):
        flag = 1
        for i, j in sents_cos[1:11]:
            if len(word_tokenize(temp_max_summ)) + len(word_tokenize(i)) < word_limit:
                flag = 0
                temp = cosine_similarity(ori_doc_embedding, model.encode(temp_max_summ + " " + i).reshape(1, -1))[0][0]
                if temp > temp_max_summ_sim:
                    temp_max_summ = temp_max_summ + " " + i
                    temp_max_summ_sim = temp
                    sents_cos.remove((i, j))
                    # print(sents_cos)
                    break

        if flag:
            break

    # print(temp_max_summ)
    max_summ = temp_max_summ
    ori_single_max_summ_cos = cosine_similarity(ori_doc_embedding, model.encode(max_summ).reshape(1, -1))[0][0]
    final_score = int(input_summary_cos * 100 / ori_single_max_summ_cos)

    if final_score < 0:
        final_score = 0

    return final_score, input_summary_cos, ori_single_max_summ_cos






# 接收POST请求数据
def score_post(request):
    ctx = {}
    if request.POST:
        # print(request.POST['summary'])
        # print(request.POST['ori_doc'])
        input_summary = request.POST['summary']
        ori_doc = request.POST['ori_doc']
        sum_length = request.POST['length']
        # print(sum_length)
        # if sum_length == "":
        #     print("sum_length is None")

        if input_summary != "" and ori_doc != "":
            final_score, input_summary_cos, ori_single_max_summ_cos = scoring(input_summary, ori_doc, sum_length)
            ctx['rlt'] = final_score
            ctx['input_summary_cos'] = input_summary_cos
            ctx['ori_single_max_summ_cos'] = ori_single_max_summ_cos

            ctx['summary'] = input_summary
            ctx['ori_doc'] = ori_doc

            if final_score>=97:
                ctx['rank'] = "A+"
            elif final_score>=93 and final_score<=96:
                ctx['rank'] = "A"
            elif final_score>=90 and final_score<=92:
                ctx['rank'] = "A-"
            elif final_score>=87 and final_score<=89:
                ctx['rank'] = "B+"
            elif final_score>=83 and final_score<=86:
                ctx['rank'] = "B"
            elif final_score>=80 and final_score<=82:
                ctx['rank'] = "B-"
            elif final_score>=77 and final_score<=79:
                ctx['rank'] = "C+"
            elif final_score>=73 and final_score<=76:
                ctx['rank'] = "C"
            elif final_score>=70 and final_score<=72:
                ctx['rank'] = "C-"
            elif final_score>=67 and final_score<=69:
                ctx['rank'] = "D+"
            elif final_score>=63 and final_score<=66:
                ctx['rank'] = "D"
            elif final_score>=60 and final_score<=62:
                ctx['rank'] = "D-"
            elif final_score<60:
                ctx['rank'] = "F"


        else:
            ctx['rlt'] = "请输入摘要和原文"
    return render(request, "post_score.html", ctx)
