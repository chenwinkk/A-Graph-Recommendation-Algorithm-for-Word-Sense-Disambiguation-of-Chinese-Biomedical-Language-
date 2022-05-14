import torch
def process_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx
def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
class Regularization(torch.nn.Module):

    def __init__(self, model, gamma=0.01, p=2, device="cpu"):
        '''
        :param model:构建好的模型
        :param gamma:系数
        :param p:当p=0表示L2正则化，p=1表示L1正则化
        '''
        super().__init__()
        if gamma <= 0:
            print("param weight_decay can not be <= 0")
            exit(0)
        self.model = model
        self.gamma = gamma
        self.p = p
        self.device = device
        self.weight_list = self.get_weight_list(model) # 取出参数的列表
        self.weight_info = self.get_weight_info(self.weight_list) # 打印参数的信息

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, model):
        self.weight_list = self.get_weight_list(model)
        reg_loss = self.regulation_loss(self.weight_list, self.gamma, self.p)
        return reg_loss

    def regulation_loss(self, weight_list, gamma, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss += l2_reg
        reg_loss = reg_loss * gamma
        return reg_loss

    def get_weight_list(self, model):
        weight_list = []
        # 返回参数的名字和参数自己
        for name, param in model.named_parameters():
            # 这里只取weight 未取bias
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def get_weight_info(self, weight_list):
        # 打印被正则化的参数的名称
        print("#"*10, "regulations weight", "#"*10)
        for name, param in weight_list:
            print(name)
        print("#"*25)