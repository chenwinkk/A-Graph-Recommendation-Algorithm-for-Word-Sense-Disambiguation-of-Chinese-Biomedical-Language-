import pandas as pd
from jieba import iteritems

from tqdm import tqdm
from collections import defaultdict
import math
import copy
import logging
import numpy as np
import pickle
#召回  用于粗选一批待推荐的词
#数据读入
train_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_train.json')
#对icd进行处理成字典
icd_df = pd.read_excel(
    '../data/source_datasets/CHIP-CDN/国际疾病分类 ICD-10北京临床版v601.xlsx',
    header=None,
    names=['icd_code', 'name']
)
#当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值，默认值为空字符
map_dict = defaultdict(set)
for _text in icd_df['name']:
    map_dict[_text].add(_text)
#打标签
for _text in train_df['normalized_result']:
    for _label in _text.split('##'):
        map_dict[_label].add(_label)
for _text, _labels in zip(train_df['text'], train_df['normalized_result']):
    for _label in _labels.split('##'):
        map_dict[_text].add(_label)
#写入日志
logger = logging.getLogger(__name__)

#召回模型构建
class BM25(object):
    """
    BM25模型

    Args:
        corpus (:obj:`list`):
            检索的语料
        k1 (:obj:`float`, optional, defaults to 1.5):
            取正值的调优参数，用于文档中的词项频率进行缩放控制
        b (:obj:`float`, optional, defaults to 0.75):
            0到1之间的参数，决定文档长度的缩放程度，b=1表示基于文档长度对词项权重进行完全的缩放，b=0表示归一化时不考虑文档长度因素
        epsilon (:obj:`float`, optional, defaults to 0.25):
            idf的下限值
        tokenizer (:obj:`object`, optional, defaults to None):
            分词器，用于对文档进行分词操作，默认为None，按字颗粒对文档进行分词
        is_retain_docs (:obj:`bool`, optional, defaults to False):
            是否保持原始文档

    """
    def __init__(
        self,
        corpus,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        tokenizer=None,
        is_retain_docs=False
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.docs = None
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = [] # 用于记录在文本中出现两次以上的词
        self.idf = {}
        self.doc_len = []

        if is_retain_docs:
            self.docs = copy.deepcopy(corpus)

        if tokenizer:
            corpus = [self.tokenizer.tokenize(document) for document in corpus]
        else:
            corpus = [list(document) for document in corpus]

        self._initialize(corpus)

    def _initialize(self, corpus):
        """计算词语在句子与语料库中的频率，同时计算IDF值"""
        nd = {}  # 存储每个词出现了该词的文档数目
        num_doc = 0 # 文档的数量
        for document in corpus:
            self.corpus_size += 1 #语料长度增加
            self.doc_len.append(len(document)) #文档长度
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1  # 同一个词出现频率
            self.doc_freqs.append(frequencies) # 出现两个或两个以上的词

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size  # 计算文档d的平均长度
        # idf值
        idf_sum = 0
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5) #计算idf
            self.idf[word] = idf  # 该词对应的idf
            idf_sum += idf #idf总和
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf) #平均idf

        if self.average_idf < 0: #如果idf小于0 报错
            logger.warning(
                'Average inverse document frequency is less than zero. Your corpus of {} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'.format(self.corpus_size)
            )

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps  # 不能低于下限值

    def get_score(self, query, index): # 计算相关性得分
        score = 0.0
        doc_freqs = self.doc_freqs[index]
        numerator_constant = self.k1 + 1
        denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
        for word in query:
            if word in doc_freqs:
                df = self.doc_freqs[index][word]
                idf = self.idf[word]
                score += (idf * df * numerator_constant) / (df + denominator_constant)
        return score

    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in range(self.corpus_size)]
        return scores

    def recall(self, query, topk=5):#召回
        scores = self.get_scores(query)
        indexs = np.argsort(scores)[::-1][:topk]  #提取前k个最大的索引

        if self.docs is None:
            return [[i, scores[i]] for i in indexs]
        else:
            return [[self.docs[i], scores[i]] for i in indexs]


bm25_model = BM25([_text for _text, _ in map_dict.items()], is_retain_docs=True)

if __name__ == "__main__":
    #召回评估
    dev_data_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_dev.json')

    a_label = []
    new_train_data = []
    recall_ = 0
    query_counter = 0
    miss_list = []

    for text_, normalized_result_ in tqdm(zip(dev_data_df['text'], dev_data_df['normalized_result'])):
        query_counter += 1
        result = set([_result for _results in bm25_model.recall(text_, topk=200) for _result in map_dict[_results[0]]])
        if len(set(normalized_result_.split('##')) & result) != len(set(normalized_result_.split('##'))):
            miss_list.append([text_, normalized_result_])
            continue

        recall_ += 1

    print('召回率为： ', recall_ / query_counter)

    with open('../checkpoint/recall/bm25_model.pkl', "wb") as f:
        pickle.dump(bm25_model, f)

    with open('../checkpoint/recall/map_dict.pkl', "wb") as f:
        pickle.dump(map_dict, f)
