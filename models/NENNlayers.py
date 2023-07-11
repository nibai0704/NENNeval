import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, in_edge_features,out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.in_edge_features = in_edge_features

        self.W_n = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_n.data, gain=1.414)
        self.a_n = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_n.data, gain=1.414)

        self.W_e = nn.Parameter(torch.empty(size=(in_edge_features, out_features)))
        nn.init.xavier_uniform_(self.W_e.data, gain=1.414)        
        self.a_e = nn.Parameter(torch.empty(size=(2*out_features, 1)))# ae的大小是节点+边的out 但这里这个也行因为我们的out的大小是一样的
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        # qe和qn都是边的可学习向量
        self.q_e = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.q_e.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.q_n = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.q_n.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)



    def forward(self, h, edge_h,adj,edge_adj,node_edge_adj):
        # 506,256;2879,1;506,506;2879,2879;2879x506
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 对节点对节点的注意力系数的计算和原本的GAT差不多
        # n对n使用的是wh_n
        # 256-8 16-16 32-32 64-64 128-128 256-1
        #print('NENNlayers')
        Wh_n = torch.mm(h, self.W_n) # 506 256 256 8 最终的506 256 64 1
        e_n_n = self._prepare_attentional_mechanism_input(Wh_n,'n_n')

        zero_vec_n_n = -9e15*torch.ones_like(e_n_n)# 506 506
        attention_n_n = torch.where(adj > 0, e_n_n, zero_vec_n_n)
        attention_n_n = F.softmax(attention_n_n, dim=1)
        attention_n_n = F.dropout(attention_n_n, self.dropout, training=self.training)
        h_prime_n_n = torch.matmul(attention_n_n, Wh_n) #506 506 506 8
        # 对边对节点的注意力系数的计算要引入可学习
        # e对n e_n 用两个边权值矩阵Wh_e 和wh_n 在挑选的时候是node_edge_adj
        Wh_e = torch.mm(edge_h, self.W_e) # 2879 1 1 8
        e_e_n = self._prepare_attentional_mechanism_input_eandn(Wh_n,Wh_e,'e_n')

        zero_vec_e_n = -9e15*torch.ones_like(e_e_n) #  506 2879
        # 下面 een2879 506 nodeedge维度反了
        attention_e_n = torch.where(node_edge_adj.T > 0, e_e_n, zero_vec_e_n)
        # 维度不一致
        attention_e_n = F.softmax(attention_e_n, dim=1)
        attention_e_n = F.dropout(attention_e_n, self.dropout, training=self.training)
        h_prime_e_n = torch.matmul(attention_e_n, Wh_e)# 506x2879 and whe 2879x8
        #下面就上面两个concat 返回作为这一层的节点嵌入

        # 边对边的注意力机制和节点对节点的类似
        e_e_e = self._prepare_attentional_mechanism_input(Wh_e,'e_e')

        zero_vec_e_e = -9e15*torch.ones_like(e_e_e)
        attention_e_e = torch.where(edge_adj > 0, e_e_e, zero_vec_e_e)
        attention_e_e = F.softmax(attention_e_e, dim=1)
        attention_e_e = F.dropout(attention_e_e, self.dropout, training=self.training)
        h_prime_e_e = torch.matmul(attention_e_e, Wh_e)# 2879x2879 2879 8      

        # 节点对边的注意力和边对节点的类似
        e_n_e = self._prepare_attentional_mechanism_input_eandn(Wh_n,Wh_e,'n_e')

        zero_vec_n_e = -9e15*torch.ones_like(e_n_e)
        attention_n_e = torch.where(node_edge_adj > 0, e_n_e, zero_vec_n_e)
        attention_n_e = F.softmax(attention_n_e, dim=1)
        attention_n_e = F.dropout(attention_n_e, self.dropout, training=self.training)
        h_prime_n_e = torch.matmul(attention_n_e, Wh_n)# 2879x506 506x8


        h_prime_n = torch.cat([h_prime_n_n,h_prime_e_n],dim=1)
        h_prime_e = torch.cat([h_prime_n_e,h_prime_e_e],dim=1)
        return F.elu(h_prime_n),F.elu(h_prime_e)

    def _prepare_attentional_mechanism_input_eandn(self, Wh_n,Wh_e,flag):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        if(flag == 'n_e'):
            Wh1 = torch.matmul(Wh_e, self.q_n[:self.out_features, :])
            Wh2 = torch.matmul(Wh_n, self.q_n[self.out_features:, :])
            # broadcast add
        if(flag == 'e_n'):
            Wh1 = torch.matmul(Wh_n, self.a_e[:self.out_features, :])
            Wh2 = torch.matmul(Wh_e, self.a_e[self.out_features:, :])
        e = Wh1 + Wh2.T# 行每一行都会复制，列同样
        return self.leakyrelu(e)
    def _prepare_attentional_mechanism_input(self, Wh,flag):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        if(flag == 'n_n'):
            Wh1 = torch.matmul(Wh, self.a_n[:self.out_features, :])
            Wh2 = torch.matmul(Wh, self.a_n[self.out_features:, :])
            # broadcast add
        if(flag == 'e_e'):
            Wh1 = torch.matmul(Wh, self.q_e[:self.out_features, :])
            Wh2 = torch.matmul(Wh, self.q_e[self.out_features:, :])
        e = Wh1 + Wh2.T# 行每一行都会复制，列同样
        return self.leakyrelu(e)
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
