import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("models/")
from NENNlayers import GraphAttentionLayer
from mlp import MLP

class GAT(nn.Module):
    def __init__(self, nfeat,efeat ,nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_att = GraphAttentionLayer(nfeat,efeat, nhid*1, dropout=dropout, alpha=alpha, concat=True)
        # in_att 256,1,8
        # nheads是4 所以下面i是0123
        self.attentions = [GraphAttentionLayer(nhid*pow(2,i+1),nhid*pow(2,i+1), nhid*pow(2,i+1), dropout=dropout, alpha=alpha, concat=True) for i in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid *pow(2,nheads+1),nhid *pow(2,nheads+1),nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = GraphAttentionLayer(64,64,nclass, dropout=dropout, alpha=alpha, concat=False)

        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        # 一层mlp带输入的，三层mlp纯隐藏层 归一化和mlp的层数是对应的，还有一层线性预测
        for layer in range(2):
            if layer == 0:
                self.mlps.append(MLP(2, 256, 64, 64))# 256是input
            else:
                self.mlps.append(MLP(2, 64, 64, 64))
            # num_features是需要归一化的那一维的维度
            self.batch_norms.append(nn.BatchNorm1d(64))
    def forward(self, x,e_x ,adj,e_adj,n_e_adj):
        #print('NENN')
        x = F.dropout(x, self.dropout, training=self.training)
        e_x = F.dropout(e_x, self.dropout, training=self.training)
        # 多个注意力头cat起来
        #x = torch.cat([att(x,e_x ,adj,e_adj,n_e_adj) for att in self.attentions], dim=1)
        x,e_x=self.in_att(x,e_x ,adj,e_adj,n_e_adj)

        for att in self.attentions:
            x,e_x = att(x,e_x ,adj,e_adj,n_e_adj)
        final_x = F.dropout(x, self.dropout, training=self.training)
        final_e_x = F.dropout(e_x, self.dropout, training=self.training)
        #print('graphCNN')# 在concat之前拿GIN聚合一下信息作测试吧，中间这一段有用就放进去没用就算了
        h_n = final_x
        h_e = final_e_x
        for layer in range(2):
            h_n = self.next_layer(h_n,layer,adj)
            h_e = self.next_layer(h_e,layer,e_adj)
       
        final_x = F.elu(self.out_att(h_n,h_e ,adj,e_adj,n_e_adj)[0])# 最后一层是不cat的 但是NENN里没说明最后      
        final_e_x = F.elu(self.out_att(h_n,h_e ,adj,e_adj,n_e_adj)[1])

        return F.log_softmax(final_x, dim=1),F.log_softmax(final_e_x, dim=1)

    def next_layer(self, h, layer, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
        # 将相邻节点和中心节点合并在一起    
        # print('h:{}\n layer:{}\n Adj_block:{}\n'.format(h,layer,Adj_block.values))
        Adj_block = Adj_block.float()
        pooled = torch.spmm(Adj_block, h)

        # pooled是x_concat和Adj_block的乘积，把它放到MLPs里面会有什么结果呢？有什么意义呢？

        pooled_rep = self.mlps[layer](pooled)
        # print('\npooled_rep:{}'.format(pooled_rep.size()))#pooled_rep:torch.Size([564, 64])
        # print('\npooled:{}'.format(pooled.size()))
        # #pooled:torch.Size([564, 7]) 
        # 是一个整数构成的较为稀疏的矩阵，由邻接矩阵和特征向量构成
        # 一共四对数据 pooled的形状除了第一次输入之外都是[564, 64]
        # BatchNorm1d用来归一化
        h = self.batch_norms[layer](pooled_rep)        
        #non-linearity
        # print('pooled:{}\n pooled_rep:{}\n h:{}\n'.format(pooled,pooled_rep,h))
        h = F.relu(h)
        # print('h_relu:{}'.format(h.size())) relu操作前后四对数据都是torch.Size([564, 64])
        return h