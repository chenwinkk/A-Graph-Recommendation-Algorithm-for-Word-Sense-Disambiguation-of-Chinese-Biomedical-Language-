import torch
import pickle
import pandas as pd

from ark_nlp.model.tm.bert import Bert
from ark_nlp.model.tm.bert import BertConfig
from ark_nlp.model.tm.bert import Dataset
from ark_nlp.model.tm.bert import Task
from ark_nlp.model.tm.bert import get_default_model_optimizer
from ark_nlp.model.tm.bert import Tokenizer
from ark_nlp.factory.predictor import TMPredictor
import math
import copy
import logging
import numpy as np

from six import iteritems
import pickle

logger = logging.getLogger(__name__)


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


path2 = "D://pythonProject6//data//source_datasets//CHIP-CDN//train"
def recall(i,sim):
    G = nx.read_edgelist(path2 + "//" + str(i) + ".txt",
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    # 得到128维的句子向量
    embeddings = model.get_embeddings()
    k = 0
    sim_em = []
    for embedding in test_embedding:
        x = cos_sim(embedding, embeddings)
        if x > sim:
            sim_em.append(text1[k])
        k = k + 1
    result = set([re for re in sim_em])
    return result
# 候选标准词匹配

if __name__ == "__main__":
    # 读入模型
    # bm25_model = pickle.load(open('../checkpoint/recall/bm25_model.pkl', 'rb'))
    # map_dict = pickle.load(open('../checkpoint/recall/map_dict.pkl', 'rb'))
    # 读入数据
    train_data_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_train.json')
    dev_data_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_dev.json')
    # 数据生成

    icd_df = pd.read_excel(
        '../data/source_datasets/CHIP-CDN/国际疾病分类 ICD-10北京临床版v601.xlsx',
        header=None,
        names=['icd_code', 'name']
    )
    text1 = []
    for text_ in icd_df['name']:
        text1.append(text_)
    path = "D://pythonProject6//data//source_datasets//CHIP-CDN//norm"
    test_embedding = []
    for j in range(1):
        G = nx.read_edgelist(path + "//" + str(j) + ".txt",
                             create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

        model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
        model.train(window_size=5, iter=3)
        # 得到128维的句子向量
        embeddings = model.get_embeddings()
        test_embedding.append(embeddings)
    # 链接诊断原词和候选标准词，构造正负样本
    pair_dataset = []
    i = 0
    for _raw_word, _normalized_result in zip(train_data_df['text'], train_data_df['normalized_result']):
        normalized_words = set(_normalized_result.split('##'))  # 标准化词
        search_result_ = set()  # 查询结果，用于存放已经查过的词
        train_pair_dataset = []  # 匹配数据
        print(recall(i,sim=0.9))
        print(_raw_word)
        print(i)
        for index,_search_word in enumerate(
            #需要修改点:把找回模型封装后，能够输入一个句子，对其召回,要改_raw_word
                [_results for _results in recall(i, sim=0.9)]):
            # 从召回的数据中进行精选，如果在标准结果中则继续
            print(_search_word)
            if _search_word in normalized_words:
                continue
            # 历史查询
            elif _search_word in search_result_:
                continue
            else:
                train_pair_dataset.append([_raw_word, _search_word, '0'])  # 如果没有则加入0分类

            search_result_.add(_search_word)
            # 精选20个词
            if len(train_pair_dataset) == 20:
                pair_dataset.extend(train_pair_dataset)
                break
        i += 1
        for _st_word in normalized_words:
            for _ in range(10):
                pair_dataset.append([_raw_word, _st_word, '1'])  # 匹配成功加入1分类

    print(pair_dataset)
    # 预测集
    pair_dev_dataset = []
    j = 0
    for _raw_word, _normalized_result in zip(train_data_df['text'], train_data_df['normalized_result']):
        normalized_words = set(_normalized_result.split('##'))
        search_result_ = set()
        dev_pair_dataset = []
        for index,_search_word in enumerate(
                [_results for _results in recall(j, 0.77) ]):

            if _search_word in normalized_words:
                continue
            elif _search_word in search_result_:
                continue
            else:
                dev_pair_dataset.append([_raw_word, _search_word, '0'])

            search_result_.add(_search_word)

            if len(dev_pair_dataset) == 1:
                pair_dev_dataset.extend(dev_pair_dataset)
                break
        j = j + 1
        for _st_word in normalized_words:
            pair_dev_dataset.append([_raw_word, _st_word, '1'])
    print(pair_dev_dataset)
    # 数据生成，text_a 就是当前的句子，text_b是另一个句子，因为有的任务需要两个两个句子，label就是标签
    train_data_df = pd.DataFrame(pair_dataset, columns=['text_a', 'text_b', 'label'])
    dev_data_df = pd.DataFrame(pair_dev_dataset, columns=['text_a', 'text_b', 'label'])
    print(train_data_df)
    print(dev_data_df)
    tm_train_dataset = Dataset(train_data_df)
    tm_dev_dataset = Dataset(dev_data_df)
    # 词典创建
    tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=50)
    # tokens转化成单个字的id（行号）
    tm_train_dataset.convert_to_ids(tokenizer)
    tm_dev_dataset.convert_to_ids(tokenizer)

    '''模型参数设置'''
    # 预训练模型，使用bert预训练，ERNIE 1．0 可以直接对先验语义知识单元进行建模，增强了模型语义表示能力，学习词与实体的表达
    config = BertConfig.from_pretrained('nghuyong/ernie-1.0',
                                        num_labels=len(tm_train_dataset.cat2id))
    torch.cuda.empty_cache()
    dl_module = Bert.from_pretrained('nghuyong/ernie-1.0',
                                     config=config)

    # 设置运行次数
    num_epoches = 1
    batch_size = 16

    # 给出网络层的名字和参数的迭代器，查看是否有池化层
    param_optimizer = list(dl_module.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # 不需要衰减的权重
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 文本匹配任务创建，使用ema在于利用滑动平均的参数来提高模型在测试数据上的健壮性，用adamw优化，最小二乘复指数拟合
    model = Task(dl_module, 'adamw', 'lsce', cuda_device=0, ema_decay=0.995)

    # 训练
    model.fit(tm_train_dataset,
              tm_dev_dataset,
              lr=3e-5,
              epochs=num_epoches,
              batch_size=batch_size,
              params=optimizer_grouped_parameters
              )
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # model.ema.store(model.module.parameters())
    # model.ema.copy_to(model.module.parameters())
    # 模型验证，文本匹配任务的预测器
    tm_predictor_instance = TMPredictor(model.module, tokenizer, tm_train_dataset.cat2id)
    # 单样本预测
    tm_predictor_instance.predict_one_sample(['胸部皮肤破裂伤', '胸部开放性损伤'], return_proba=True)
    # 模型保存
    torch.save(model.module.state_dict(), '../checkpoint/textsim/module.pth')
    with open('../checkpoint/textsim/cat2id.pkl', "wb") as f:
        pickle.dump(tm_train_dataset.cat2id, f)
