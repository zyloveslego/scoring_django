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
def scoring(input_summary, ori_doc):
    # input_summary = ""
    # ori_doc = ""

    # ori_doc_sents = []

    # # 取原文
    # file = open("/Users/zhouyou/Documents/git/test/summary_scoring_test/ori_doc.txt")
    #
    # for i in file.readlines():
    #     ori_doc += i.replace("\n", "") + " "
    #     ori_doc_sents.append(i)
    #
    # file.close()
    #
    # # 取摘要
    # file = open("/Users/zhouyou/Documents/git/test/summary_scoring_test/input_summary.txt")
    #
    # for i in file.readlines():
    #     input_summary += i.replace("\n", "") + " "
    #
    # file.close()

    ori_doc_sents = sent_tokenize(ori_doc)

    ori_doc_embedding = model.encode(ori_doc).reshape(1, -1)

    input_summary_embedding = model.encode(input_summary).reshape(1, -1)

    input_summary_len = len(word_tokenize(input_summary))

    ori_single_max_summ_cos = 0  # g*

    for i in range(3, 10):
        print(i)
        flag = 0
        for sents in itertools.combinations(ori_doc_sents, i):
            word_limit = int(input_summary_len) + 5
            word_count = 0
            temp_single_summ = ""
            # print(sents)
            # print("--------------")
            for ele in sents:
                # print(ele)
                text_tokens = word_tokenize(ele)
                # print(b)
                tokens_without_sw = [word for word in text_tokens if word not in string.punctuation]
                # print(tokens_without_sw)
                # print(len(tokens_without_sw))

                word_count = word_count + len(tokens_without_sw)
                temp_single_summ = temp_single_summ + " " + ele


            if word_count <= word_limit:
                flag = 1

            if int(input_summary_len * 0.8) + 2 < word_count <= int(input_summary_len) + 2:
                # print(temp_single_summ)

                temp_single_summ_embedding = model.encode(temp_single_summ).reshape(1, -1)
                ori_temp_single_summ_cos = cosine_similarity(ori_doc_embedding, temp_single_summ_embedding)[0][0]
                # ori_single_summ_cos.append(ori_temp_single_summ_cos)
                # print(ori_temp_single_summ_cos)

                if ori_temp_single_summ_cos > ori_single_max_summ_cos:
                    ori_single_max_summ_cos = ori_temp_single_summ_cos
                    # print(temp_single_summ)
                    # print(ori_single_max_summ_cos)

            # break

        if flag == 0:
            print("over")
            break

        # break

    input_summary_cos = cosine_similarity(ori_doc_embedding, input_summary_embedding)[0][0]

    # print(input_summary)
    print("输入的相似度:")
    print(input_summary_cos)
    print("最好的相似度:")
    print(ori_single_max_summ_cos)

    final_score = int(input_summary_cos * 100 / ori_single_max_summ_cos)
    if final_score < 0:
        final_score = 0

    print(final_score)

    return final_score


# 接收POST请求数据
def score_post(request):
    ctx = {}
    if request.POST:
        print(request.POST['summary'])
        print(request.POST['ori_doc'])
        input_summary = request.POST['summary']
        ori_doc = request.POST['ori_doc']

        final_score = scoring(input_summary, ori_doc)

        ctx['rlt'] = final_score
    return render(request, "post_score.html", ctx)
