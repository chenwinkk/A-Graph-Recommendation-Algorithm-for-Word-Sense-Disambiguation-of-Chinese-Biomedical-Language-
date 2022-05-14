# coding = gbk

import math
import copy
import logging
import numpy as np
import pickle
import torch
import pandas as pd
import json
from six import iteritems
from ark_nlp.model.tc.bert import Bert as TCBert
from ark_nlp.model.tc.bert import BertConfig as TCBertConfig
from ark_nlp.model.tc.bert import Dataset as TCDataset
from ark_nlp.model.tc.bert import Tokenizer as TCTokenizer
from ark_nlp.factory.predictor import TCPredictor
from ark_nlp.model.tm.bert import Bert as TMBert
from ark_nlp.model.tm.bert import BertConfig as TMBertConfig
from ark_nlp.model.tm.bert import Dataset as TMDataset
from ark_nlp.model.tm.bert import Task as TMTask
from ark_nlp.model.tm.bert import Tokenizer as TMTokenizer
from ark_nlp.factory.predictor import TMPredictor
from ark_nlp.dataset.base._dataset import BaseDataset
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from graphviz import Digraph
import sys
import json


logger = logging.getLogger(__name__)


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


    """  # noqa: ignore flake8"

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
        self.doc_freqs = []
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
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size

        idf_sum = 0
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        if self.average_idf < 0:
            logger.warning(
                'Average inverse document frequency is less than zero. Your corpus of {} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'.format(self.corpus_size)
            )

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, query, index):
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

    def recall(self, query, topk=5):
        scores = self.get_scores(query)
        indexs = np.argsort(scores)[::-1][:topk]

        if self.docs is None:
            return [[i, scores[i]] for i in indexs]
        else:
            return [[self.docs[i], scores[i]] for i in indexs]

'''疾病标准化'''
class PCTestDataset(BaseDataset):

    def _get_categories(self):
        return ''

    def _convert_to_dataset(self, data_df):

        dataset = []
        # 归一化
        data_df['text_a'] = data_df['text_a'].apply(lambda x: x.lower().strip())
        data_df['text_b'] = data_df['text_b'].apply(lambda x: x.lower().strip())
        # 得到特征
        feature_names = list(data_df.columns)
        # 返回三元组
        for index_, row_ in enumerate(data_df.itertuples()):
            dataset.append({feature_name_: getattr(row_, feature_name_)
                            for feature_name_ in feature_names})

        return dataset

    def _convert_to_transfomer_ids(self, bert_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])

            input_ids, input_mask, segment_ids = input_ids

            input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)

            feature = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': segment_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features

    def _convert_to_vanilla_ids(self, vanilla_tokenizer):

        features = []
        for (index_, row_) in enumerate(self.dataset):

            input_a_ids = vanilla_tokenizer.sequence_to_ids(row_['text_a'])
            input_b_ids = vanilla_tokenizer.sequence_to_ids(row_['text_b'])

            feature = {
                'input_a_ids': input_a_ids,
                'input_b_ids': input_b_ids
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                feature['label_ids'] = label_ids

            features.append(feature)

        return features


def get_operation_icd_name_batch(query_name):
    predict_num = tc_predictor_instance.predict_one_sample(query_name)[0] # 得到预测数量

    result = []
    search_set = set()
    batch_list = []

    for _index, _search_word in enumerate(
            [_result for _results in bm25_model.recall(query_name, topk=1000) for _result in map_dict[_results[0]]]):

        if _search_word not in search_set:
            batch_list.append([query_name, _search_word])
            search_set.add(_search_word)

        if _index == 200:
            break

    if len(batch_list) == 1:
        batch_list = [batch_list]

    batch_df = pd.DataFrame(batch_list, columns=['text_a', 'text_b'])

    batch_dataset = PCTestDataset(batch_df, is_test=True)
    batch_dataset.convert_to_ids(predict_textsim_tokenizer)
    batch_predict_ = tm_predictor_instance.predict_batch(batch_dataset, return_proba=True)

    statistics = []
    for (query_name, recall_result_), predict_ in zip(batch_list, batch_predict_):
        if predict_[0] == "1":
            statistics.append(predict_[-1])
            result.append([recall_result_, predict_[0], predict_[-1]])

    if len(result) == 0:
        for (query_name, recall_result_), predict_ in zip(batch_list, batch_predict_):
            if predict_[-1] > np.median(statistics):
                result.append([recall_result_, predict_[0], 1 - predict_[-1]])

    result = sorted(result, key=lambda x: x[-1], reverse=True)

    if len(result) == 0:
        return ''

    if predict_num == '1':
        return result[0][0]
    elif predict_num == '2':
        if len(result) >= 2:
            return result[0][0] + '##' + result[1][0]
        else:
            return result[0][0]
    else:
        st_word_ = ''
        for index_, word_ in enumerate(result):
            if word_[-1] > 0.8:
                st_word_ += word_[0]
                if index_ != len(result) - 1:
                    st_word_ += '##'

            if index_ > 5:
                break

        if st_word_ == '':
            st_word_ = result[0][0]

    if st_word_[-1] == '#':
        return st_word_[:-2]

    return st_word_

#依赖树分析
def dependency(sentence, language='zh'):
    """
    使用Stanfordcorenlp构建依存树
    :param sentence: 需要构建树的句子
    :param language: 支持的语言
    :return: 依存序列
    """
    nlp_zh = StanfordCoreNLP(r'D:\study\GRADUATE\stanford-corenlp-full-2018-01-31', lang='zh')  # 导入解压之后的路径
    words = list(nlp_zh.word_tokenize(sentence))  # 分词
    depend = list(nlp_zh.dependency_parse(sentence))  # 依存关系
    return depend, words

def Dependency_tree_visualization(dependency_tree, words):
    """
    将依存序列与对应的单词进行匹配，并将依存树可视化，最终将依存树图片保存为jpg
    :param dependency_tree: 依存树序列
    :param words: 经过
    :return:
    """
    dependency_tree.sort(key=lambda x: x[2])  # 排序
    words = [w + "-" + str(idx) for idx, w in enumerate(words)]
    rely_id = [arc[1] for arc in dependency_tree]  # 依存ID
    relation = [arc[0] for arc in dependency_tree]  # 依存语法
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]
    # 输出匹配单词的依存树
    '''
    for i in range(len(words)):
        t = relation[i] + '(' + words[i] + ', ' + heads[i] + ')'
        print(json.dumps({
        "params": t
        }))
    '''
    # 将依存树保存为jpg图片
    g = Digraph("Dependency_tree", format="jpg")
    # 节点定义
    g.node(name='Root', fontname="SimSun", shape='doublecircle')
    for word in words:
        g.node(name=word, fontname="SimSun", label=word.split("-")[0])
    # 设置图节点
    for i in range(len(words)):
        if relation[i] not in ['HED']:
            g.edge(heads[i], words[i], label=relation[i])
        else:
            if heads[i] == 'Root':
                g.edge('Root', words[i], label=relation[i])
            else:
                g.edge('Root', heads[i], label=relation[i])

    g.render(cleanup=True)

if __name__ == "__main__":

    bm25_model = pickle.load(open('D:/pythonProject6/checkpoint/recall/bm25_model.pkl', 'rb'))
    map_dict = pickle.load(open('D:/pythonProject6/checkpoint/recall/map_dict.pkl', 'rb'))
    # 个数预测模型
    model_class = 'nghuyong/ernie-1.0'
    predict_num_model_path = 'D:/pythonProject6/checkpoint/predict_num/module.pth'
    predict_num_cat2id_path = 'D:/pythonProject6/checkpoint/predict_num/cat2id.pkl'

    predict_num_tokenizer = TCTokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=100)

    with open(predict_num_cat2id_path, "rb") as f:
        predict_num_cat2id = pickle.load(f)

    predict_num_bert_config = TCBertConfig.from_pretrained(
        model_class,
        num_labels=len(predict_num_cat2id)
    )

    predict_num_dl_module = TCBert(config=predict_num_bert_config)

    predict_num_dl_module.load_state_dict(torch.load(predict_num_model_path, map_location='cuda:0'))
    predict_num_dl_module = predict_num_dl_module.eval()

    tc_predictor_instance = TCPredictor(predict_num_dl_module, predict_num_tokenizer, predict_num_cat2id)

    # 文本相似度模型
    model_class = 'nghuyong/ernie-1.0'
    textsim_model_path = 'D:/pythonProject6/checkpoint/textsim/module.pth'
    textsim_cat2id_path = 'D:/pythonProject6/checkpoint/textsim/cat2id.pkl'

    predict_textsim_tokenizer = TMTokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=100)

    with open(textsim_cat2id_path, "rb") as f:
        predict_textsim_cat2id = pickle.load(f)

    bert_config = TMBertConfig.from_pretrained(model_class,
                                               num_labels=len(predict_textsim_cat2id))

    predict_textsim_module = TMBert(config=bert_config).to('cuda:0')
    predict_textsim_module.load_state_dict(torch.load(textsim_model_path))
    predict_textsim_module = predict_textsim_module.eval()

    tm_predictor_instance = TMPredictor(predict_textsim_module, predict_textsim_tokenizer, predict_textsim_cat2id)

    test_data_df = pd.read_json('D:/pythonProject6/data/source_datasets/CHIP-CDN/CHIP-CDN_test.json')

    submit = []
    i = 0
    j=0
    text = []
    # value = sys.argv[1]
    # params = json.loads(value)
        #text = input("请输入：")

        #text.append(params)
        #dict转成str

        #标准化
    # predict_ = get_operation_icd_name_batch(params)
    '''
    submit.append({
            #'text':params,
        'text':params,
        'normalized_result':predict_
    })
'''
    # de_line, word = dependency(params)
    # Dependency_tree_visualization(de_line, word)
    '''
    print(de_line)
    print(word)
    print(submit[i])
'''
    #传给前端
    # print(json.dumps({
    #     'word': word,
    #     "code": 0 ,
    #     "params": predict_,
    # }))
    for text_ in tqdm(test_data_df['text'].to_list()):
        predict_ = get_operation_icd_name_batch(text_)
        submit.append({
            'text': text_,
            'normalized_result': predict_
        })
    output_path = '../data/output_datasets/CHIP-CDN_test.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(submit, ensure_ascii=False))