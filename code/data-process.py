# 处理成nx能够处理的图数据
from stanfordcorenlp import StanfordCoreNLP
import torch
import pandas as pd
import numpy as np

import numpy as np
from tqdm import tqdm

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
import pickle

nlp = StanfordCoreNLP(r'D:\study\GRADUATE\stanford-corenlp-full-2018-01-31', lang='zh')


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def data2tree(sentence):
    dep_outputs = nlp.dependency_parse(sentence)
    tokens = nlp.word_tokenize(sentence)
    # 查找根节点的索引
    root_index = []
    for i in range(len(dep_outputs)):
        if dep_outputs[i][0] == 'ROOT':
            root_index.append(i)
    # 修改依存关系成为三元组
    new_dep_outputs = []
    # 对一句话中嵌套的多个树进行拆分  即对每个节点都加上上一个树的节点总数
    for i in range(len(dep_outputs)):
        for index in root_index:
            if i + 1 > index:
                tag = index
            if dep_outputs[i][0] == 'ROOT':
                dep_output = (dep_outputs[i][0], dep_outputs[i][1], dep_outputs[i][2] + tag)
            else:
                dep_output = (dep_outputs[i][0], dep_outputs[i][1] + tag, dep_outputs[i][2])
        new_dep_outputs.append(dep_output)

    return new_dep_outputs


def tree2txt(new_dep_outputs, i):
    path = "D://pythonProject6//data//source_datasets//CHIP-CDN//test"
    f = open(path + "//" + str(i) + ".txt", "wb")
    for line in new_dep_outputs:
        a = line[1]
        b = line[2]
        f.write(str(a).encode())
        f.write("  ".encode())
        f.write(str(b).encode())
        f.write('\r\n'.encode())
    f.close()


if __name__ == '__main__':
    icd_df = pd.read_excel(
        '../data/source_datasets/CHIP-CDN/国际疾病分类 ICD-10北京临床版v601.xlsx',
        header=None,
        names=['icd_code', 'name']
    )
    text1 = []
    for text_ in icd_df['name']:
        text1.append(text_)
    train_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_train.json')
    text2 = []
    text3 = []
    for text_ in train_df['text']:
        text2.append(text_)
    for text_ in train_df['normalized_result']:
        text3.append(text_)
    # print(set(text3[0].split('##')))
    # print(set(text3[0].split('##'))&set(text1[18426].split(' ')))
    # print(len(set(text3[0].split('##')) | set(text1[18426].split(' '))))
    # print(len(set(text3[0].split('##'))))
    path = "D://pythonProject6//data//source_datasets//CHIP-CDN//norm"
    test_embedding = []
    for j in range(len(text1)):
        G = nx.read_edgelist(path + "//" + str(j) + ".txt",
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=5, iter=3)
        # 得到128维的句子向量
        embeddings = model.get_embeddings()
        test_embedding.append(embeddings)
    train_embedding = []
    path2 = "D://pythonProject6//data//source_datasets//CHIP-CDN//train"
    for j in range(299):
        G = nx.read_edgelist(path2 + "//" + str(j) + ".txt",
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=5, iter=3)
        # 得到128维的句子向量
        embeddings = model.get_embeddings()
        train_embedding.append(embeddings)

    # 计算余弦相似度
    k = 0  # 记录norm到第几个文件夹了
    l = 0  # 记录train到第几个文件夹
    query_counter = 0  # 计算请求次数
    recall_ = 0  # 召回次数
    sim_em = []

    for t_embedding in train_embedding:
        query_counter += 1
        for embedding in test_embedding:
            sim = cos_sim(embedding, t_embedding)
            if sim > 0.77:
                sim_em.append(text1[k])
            k += 1
        result = set([re for re in sim_em])
        if len(set(text3[l].split('##')) & result) == len(set(text3[l].split('##'))):
            recall_ += 1
        k = 0
        sim_em = []
    print("召回率：", recall_ / query_counter)
