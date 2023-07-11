from keyword_sim_411 import pass_data
from models.NENN import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import glob
import time
import random
import argparse
# 训练设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# 加载数据
node_features,edge_features,node_adj,edge_adj,node_edge_adj = pass_data()

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

model = GAT(nfeat=256,# 节点特征
            efeat = 1,#节点特征 这个是新加的
            nhid=8, # 隐藏单元数
            nclass= 1, # 之后分类用的类符
            dropout=0.6, 
            nheads=4, # 注意力头数量
            alpha=0.2) #relu的参数

model.eval()
#print([i for i in model.modules()])
#to(device)AttributeError: 'tuple' object has no attribute 'to
output = model(node_features,edge_features,torch.LongTensor(node_adj) ,torch.LongTensor(edge_adj),torch.LongTensor(node_edge_adj.T))
print(output)