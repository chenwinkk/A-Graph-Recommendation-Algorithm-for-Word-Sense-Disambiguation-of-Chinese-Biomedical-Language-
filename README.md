# A-Graph-Recommendation-Algorithm-for-Word-Sense-Disambiguation-of-Chinese-Biomedical-Language-
Code for A Graph Recommendation Algorithm for Word Sense Disambiguation of Chinese Biomedical Language 
## 运行环境
python == 3.9  
pytorch == 1.10.2  
tensorflow == 2.x  

## 运行方法
首先解压 ../data/source_datasets/CHIP-CDN/norm.zip以及train.zip  
再运行 ../code/Recall.py  模型保存于../checkpoint/recall中  
再运行 ../code/PredictTextsim.py  
在该文件中内置了DeepWalk、Node2Vec、SDNE、LINE、Struc2Vec五种Graph Embedding方法，可依照需求使用  
再运行 ../code/PredictNum.py 模型保存于../checkpoint/predict_num中  
最后运行 ../code/Standarlization.py进行标准化  
最终结果保存于 ../data/output_datasets中  
## 落地项目应用
在../code/Standarlization.py中我们同时实现了于Nodejs交互的接口，可以直接调用  
并与设计好的前后端进行连接，可以直接作为算法后端实用  
具体代码如下：
```python
value = sys.argv[1] 
params = json.loads(value) 
predict_ = get_operation_icd_name_batch(params)
