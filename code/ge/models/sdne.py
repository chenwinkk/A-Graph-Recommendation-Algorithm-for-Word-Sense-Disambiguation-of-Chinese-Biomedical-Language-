# # -*- coding:utf-8 -*-
#
# """
#
#
#
# Author:
#
#     Weichen Shen,wcshen1994@163.com
#
#
#
# Reference:
#
#     [1] Wang D, Cui P, Zhu W. Structural deep network embedding[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016: 1225-1234.(https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)
#
#
#
# """
# import time
#
# import numpy as np
# import scipy.sparse as sp
# import tensorflow
# import tensorflow as tf
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.callbacks import History
# from tensorflow.python.keras.layers import Dense, Input
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.regularizers import l1_l2
#
# from ..utils import preprocess_nxgraph
#
# def l_2nd(beta):
#     def loss_2nd(y_true, y_pred):
#         with tf.compat.v1.Session() as sess:
#             sess.run(tf.compat.v1.global_variables_initializer())
#             y_true = sess.run(y_true)
#         b_ = np.ones_like(y_true)
#         b_[y_true != 0] = beta
#         x = K.square((y_true - y_pred) * b_)
#         t = K.sum(x, axis=-1, )
#         return K.mean(t)
#
#     return loss_2nd
#
#
# def l_1st(alpha):
#     def loss_1st(y_true, y_pred):
#         L = y_true
#         Y = y_pred
#         batch_size = tf.to_float(K.shape(L)[0])
#         return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size
#
#     return loss_1st
#
#
# def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
#     A = Input(shape=(node_size,))
#     L = Input(shape=(None,))
#     fc = A
#     for i in range(len(hidden_size)):
#         if i == len(hidden_size) - 1:
#             fc = Dense(hidden_size[i], activation='relu',
#                        kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
#         else:
#             fc = Dense(hidden_size[i], activation='relu',
#                        kernel_regularizer=l1_l2(l1, l2))(fc)
#     Y = fc
#     for i in reversed(range(len(hidden_size) - 1)):
#         fc = Dense(hidden_size[i], activation='relu',
#                    kernel_regularizer=l1_l2(l1, l2))(fc)
#
#     A_ = Dense(node_size, 'relu', name='2nd')(fc)
#     model = Model(inputs=[A, L], outputs=[A_, Y])
#     emb = Model(inputs=A, outputs=Y)
#     return model, emb
#
#
# class SDNE(object):
#     def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4, ):
#
#         self.graph = graph
#         # self.g.remove_edges_from(self.g.selfloop_edges())
#         self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
#
#         self.node_size = self.graph.number_of_nodes()
#         self.hidden_size = hidden_size
#         self.alpha = alpha
#         self.beta = beta
#         self.nu1 = nu1
#         self.nu2 = nu2
#
#         self.A, self.L = self._create_A_L(
#             self.graph, self.node2idx)  # Adj Matrix,L Matrix
#         self.reset_model()
#         self.inputs = [self.A, self.L]
#         self._embeddings = {}
#
#     def reset_model(self, opt='adam'):
#
#         self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1,
#                                                   l2=self.nu2)
#         self.model.compile(opt, [l_2nd(self.beta), l_1st(self.alpha)])
#         self.get_embeddings()
#
#     def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
#         if batch_size >= self.node_size:
#             if batch_size > self.node_size:
#                 print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
#                     batch_size, self.node_size))
#                 batch_size = self.node_size
#             return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
#                                   batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose,
#                                   shuffle=False, )
#         else:
#             steps_per_epoch = (self.node_size - 1) // batch_size + 1
#             hist = History()
#             hist.on_train_begin()
#             logs = {}
#             for epoch in range(initial_epoch, epochs):
#                 start_time = time.time()
#                 losses = np.zeros(3)
#                 for i in range(steps_per_epoch):
#                     index = np.arange(
#                         i * batch_size, min((i + 1) * batch_size, self.node_size))
#                     A_train = self.A[index, :].todense()
#                     L_mat_train = self.L[index][:, index].todense()
#                     inp = [A_train, L_mat_train]
#                     batch_losses = self.model.train_on_batch(inp, inp)
#                     losses += batch_losses
#                 losses = losses / steps_per_epoch
#
#                 logs['loss'] = losses[0]
#                 logs['2nd_loss'] = losses[1]
#                 logs['1st_loss'] = losses[2]
#                 epoch_time = int(time.time() - start_time)
#                 hist.on_epoch_end(epoch, logs)
#                 if verbose > 0:
#                     print('Epoch {0}/{1}'.format(epoch + 1, epochs))
#                     print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(
#                         epoch_time, losses[0], losses[1], losses[2]))
#             return hist
#
#     def evaluate(self, ):
#         return self.model.evaluate(x=self.inputs, y=self.inputs, batch_size=self.node_size)
#
#     def get_embeddings(self):
#         self._embeddings = {}
#         doc_embeddings = []
#         embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
#         look_back = self.idx2node
#         for i, embedding in enumerate(embeddings):
#             self._embeddings[look_back[i]] = embedding
#             doc_embeddings.append(self._embeddings[look_back[i]])
#         doc = []
#         for i in range(128):
#             doc.append(0)
#         for i in range(len(doc_embeddings)):
#             doc = doc + doc_embeddings[i]
#         doc_last = doc / len(doc_embeddings)
#         return doc_last
#
#     def _create_A_L(self, graph, node2idx):
#         node_size = graph.number_of_nodes()
#         A_data = []
#         A_row_index = []
#         A_col_index = []
#
#         for edge in graph.edges():
#             v1, v2 = edge
#             edge_weight = graph[v1][v2].get('weight', 1)
#
#             A_data.append(edge_weight)
#             A_row_index.append(node2idx[v1])
#             A_col_index.append(node2idx[v2])
#
#         A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
#         A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
#                            shape=(node_size, node_size))
#
#         D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
#         L = D - A_
#         return A, L
import torch
from .basemodel import GraphBaseModel
from ..utils import process_nxgraph
import numpy as np
import scipy.sparse as sparse
from ..utils import Regularization


class SDNEModel(torch.nn.Module):

    def __init__(self, input_dim, hidden_layers, alpha, beta, device="cpu"):
        '''
        Structural Deep Network Embedding（SDNE）
        :param input_dim: 节点数量 node_size
        :param hidden_layers: AutoEncoder中间层数
        :param alpha: 对于1st_loss的系数
        :param beta: 对于2nd_loss中对非0项的惩罚
        :param device:
        '''
        super(SDNEModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        input_dim_copy = input_dim
        layers = []
        for layer_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for layer_dim in reversed(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            input_dim = layer_dim
        # 最后加一层输入的维度
        layers.append(torch.nn.Linear(input_dim, input_dim_copy))
        layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*layers)
        # torch中的只对weight进行正则真难搞啊
        # self.regularize = Regularization(self.encoder, weight_decay=gamma).to(self.device) + Regularization(self.decoder,weight_decay=gamma).to(self.device)


    def forward(self, A, L):
        '''
        输入节点的领接矩阵和拉普拉斯矩阵，主要计算方式参考论文
        :param A: adjacency_matrix, dim=(m, n)
        :param L: laplace_matrix, dim=(m, m)
        :return:
        '''
        Y = self.encoder(A)
        A_hat = self.decoder(Y)
        # loss_2nd 二阶相似度损失函数
        beta_matrix = torch.ones_like(A)
        mask = A != 0
        beta_matrix[mask] = self.beta
        loss_2nd = torch.mean(torch.sum(torch.pow((A - A_hat) * beta_matrix, 2), dim=1))
        # loss_1st 一阶相似度损失函数 论文公式(9) alpha * 2 *tr(Y^T L Y)
        loss_1st =  self.alpha * 2 * torch.trace(torch.matmul(torch.matmul(Y.transpose(0,1), L), Y))
        return loss_2nd + loss_1st




class SDNE(GraphBaseModel):

    def __init__(self, graph, hidden_layers=None, alpha=1e-5, beta=5, gamma=1e-5, device="cpu"):
        super().__init__()
        self.graph = graph
        self.idx2node, self.node2idx = process_nxgraph(graph)
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.sdne = SDNEModel(self.node_size, hidden_layers, alpha, beta)
        self.device = device
        self.embeddings = {}
        self.gamma = gamma

        adjacency_matrix, laplace_matrix = self.__create_adjacency_laplace_matrix()
        self.adjacency_matrix = torch.from_numpy(adjacency_matrix.toarray()).float().to(self.device)
        self.laplace_matrix = torch.from_numpy(laplace_matrix.toarray()).float().to(self.device)

    def fit(self, batch_size=512, epochs=1, initial_epoch=0, verbose=1):
        num_samples = self.node_size
        self.sdne.to(self.device)
        optimizer = torch.optim.Adam(self.sdne.parameters())
        if self.gamma:
            regularization = Regularization(self.sdne, gamma=self.gamma)
        if batch_size >= self.node_size:
            batch_size = self.node_size
            print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                batch_size, self.node_size))
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                optimizer.zero_grad()
                loss = self.sdne(self.adjacency_matrix, self.laplace_matrix)
                if self.gamma:
                    reg_loss = regularization(self.sdne)
                    # print("reg_loss:", reg_loss.item(), reg_loss.requires_grad)
                    loss = loss + reg_loss
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()
                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4), epoch+1, epochs))
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            for epoch in range(initial_epoch, epochs):
                loss_epoch = 0
                for i in range(steps_per_epoch):
                    idx = np.arange(i * batch_size, min((i+1) * batch_size, self.node_size))
                    A_train = self.adjacency_matrix[idx, :]
                    L_train = self.laplace_matrix[idx][:,idx]
                    # print(A_train.shape, L_train.shape)
                    optimizer.zero_grad()
                    loss = self.sdne(A_train, L_train)
                    loss_epoch += loss.item()
                    loss.backward()
                    optimizer.step()

                if verbose > 0:
                    print('Epoch {0}, loss {1} . >>> Epoch {2}/{3}'.format(epoch + 1, round(loss_epoch / num_samples, 4),
                                                                         epoch + 1, epochs))

    def get_embeddings(self):
        if not self.embeddings:
            self.__get_embeddings()
        embeddings = self.embeddings
        return embeddings

    def __get_embeddings(self):
        embeddings = {}
        with torch.no_grad():
            self.sdne.eval()
            embed = self.sdne.encoder(self.adjacency_matrix)
            doc_embeddings = []
            for i, embedding in enumerate(embed.numpy()):
                embeddings[self.idx2node[i]] = embedding
                doc_embeddings.append(embeddings[self.idx2node[i]])
            doc = []
            for i in range(128):
                doc.append(0)
            for i in range(len(doc_embeddings)):
                doc = doc + doc_embeddings[i]
            doc_last = doc / len(doc_embeddings)
        self.embeddings = doc_last


    def __create_adjacency_laplace_matrix(self):
        node_size = self.node_size
        node2idx = self.node2idx
        adjacency_matrix_data = []
        adjacency_matrix_row_index = []
        adjacency_matrix_col_index = []
        for edge in self.graph.edges():
            v1, v2 = edge
            edge_weight = self.graph[v1][v2].get("weight", 1.0)
            adjacency_matrix_data.append(edge_weight)
            adjacency_matrix_row_index.append(node2idx[v1])
            adjacency_matrix_col_index.append(node2idx[v2])
        adjacency_matrix = sparse.csr_matrix((adjacency_matrix_data,
                                              (adjacency_matrix_row_index, adjacency_matrix_col_index)),
                                             shape=(node_size, node_size))
        # L = D - A  有向图的度等于出度和入度之和; 无向图的领接矩阵是对称的，没有出入度之分直接为每行之和
        # 计算度数
        adjacency_matrix_ = sparse.csr_matrix((adjacency_matrix_data+adjacency_matrix_data,
                                               (adjacency_matrix_row_index+adjacency_matrix_col_index,
                                                adjacency_matrix_col_index+adjacency_matrix_row_index)),
                                              shape=(node_size, node_size))
        degree_matrix = sparse.diags(adjacency_matrix_.sum(axis=1).flatten().tolist()[0])
        laplace_matrix = degree_matrix - adjacency_matrix_
        return adjacency_matrix, laplace_matrix
