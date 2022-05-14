import os
import jieba
import torch
import pickle
import pandas as pd
import torch.nn as nn

from ark_nlp.model.tc.bert import Bert
from ark_nlp.model.tc.bert import BertConfig
from ark_nlp.model.tc.bert import Dataset
from ark_nlp.model.tc.bert import Task
from ark_nlp.model.tc.bert import get_default_model_optimizer
from ark_nlp.model.tc.bert import Tokenizer
from ark_nlp.factory.predictor import TCPredictor
import pickle

'''个数预测，针对一对多问题'''
#数据读入
train_data_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_train.json')
dev_data_df = pd.read_json('../data/source_datasets/CHIP-CDN/CHIP-CDN_dev.json')

train_data_df['normalized_result_num'] = train_data_df['normalized_result'].apply(lambda x: len(x.split('##'))) # 获得某个词对应标准词的数量
dev_data_df['normalized_result_num'] = dev_data_df['normalized_result'].apply(lambda x: len(x.split('##')))

train_data_df['normalized_result_num_label'] = train_data_df['normalized_result_num'].apply(lambda x: 0 if x > 2 else x)  # 打标签，如果大于2则打成0 否则直接用原数据
dev_data_df['normalized_result_num_label'] = dev_data_df['normalized_result_num'].apply(lambda x: 0 if x > 2 else x)

train_data_df = (train_data_df
                 .loc[:,['text', 'normalized_result_num_label']]
                 .rename(columns={'normalized_result_num_label': 'label'}))

dev_data_df = (dev_data_df
               .loc[:,['text', 'normalized_result_num_label']]
               .rename(columns={'normalized_result_num_label': 'label'}))
#用于序列分类的dataset
tc_train_dataset = Dataset(train_data_df)
tc_dev_dataset = Dataset(dev_data_df)

tokenizer = Tokenizer(vocab='nghuyong/ernie-1.0', max_seq_len=100)

tc_train_dataset.convert_to_ids(tokenizer)
tc_dev_dataset.convert_to_ids(tokenizer)

#预训练
config = BertConfig.from_pretrained('nghuyong/ernie-1.0',
                                    num_labels=len(tc_train_dataset.cat2id))

torch.cuda.empty_cache()

dl_module = Bert.from_pretrained('nghuyong/ernie-1.0',
                                 config=config)

if __name__ == "__main__":
    # 设置运行次数
    num_epoches = 1
    batch_size = 8

    param_optimizer = list(dl_module.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    model = Task(dl_module, 'adamw', 'lsce', cuda_device=0, ema_decay=0.995)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.fit(tc_train_dataset,
              tc_dev_dataset,
              lr=3e-5,
              epochs=num_epoches,
              batch_size=batch_size,
              params=optimizer_grouped_parameters
              )

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    #model.ema.store(model.module.parameters())

    #model.ema.copy_to(model.module.parameters())
    #文本分类任务预测器TC
    tc_predictor_instance = TCPredictor(model.module, tokenizer, tc_train_dataset.cat2id)
    tc_predictor_instance.predict_one_sample('怀孕伴精神障碍',
                                             return_proba=True)

    torch.save(model.module.state_dict(),
               '../checkpoint/predict_num/module.pth')
    with open('../checkpoint/predict_num/cat2id.pkl', "wb") as f:
        pickle.dump(tc_train_dataset.cat2id, f)

