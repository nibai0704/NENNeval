import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP

from layers import GraphAttentionLayer

''' 是否存在原则上与WL测试一样强大的GNN
    在定理3中我们的答案是肯定的如果邻居聚集和图级读出函数是内射的
    那么得到的GNN与WL测试一样强大。
'''
class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, node_input_dim, node_hidden_dim, node_output_dim, 
                 final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device,
                 nfeat,ef_sz, nhid,nclass, dropout, alpha, nheads
                 ):
        '''
            5 num_layers: number of layers in the neural networks (INCLUDING the input layer)
            2 num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            7该项会随数据维数而改变 input_dim: dimensionality of input features
            64 hidden_dim: dimensionality of hidden units at ALL layers
            2该项会随数据维数而改变 output_dim: number of classes for prediction
            0.5 final_dropout: dropout ratio on the final linear layer
            F learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            sum neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            sum graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            cuda0 device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        # 一层mlp带输入的，三层mlp纯隐藏层 归一化和mlp的层数是对应的，还有一层线性预测
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, node_input_dim, node_hidden_dim, node_hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, node_hidden_dim, node_hidden_dim, node_hidden_dim))
            # num_features是需要归一化的那一维的维度
            self.batch_norms.append(nn.BatchNorm1d(node_hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(node_input_dim, node_output_dim))
            else:
                self.linears_prediction.append(nn.Linear(node_hidden_dim, node_output_dim))
                self.dropout = dropout
        
        #起始层
        self.attentions = [GraphAttentionLayer(nfeat, nhid[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        # #hidden层
        # self.hidden_atts=[GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nhid[1], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads[1])]
        # for i, attention in enumerate(self.hidden_atts):
        #     self.add_module('hidden_att_{}'.format(i), attention)
        
        #输出层
        self.out_att = GraphAttentionLayer(nhid[0]*nheads[0]*ef_sz[0], nclass, dropout=dropout, alpha=alpha, concat=False)


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix创建块对角稀疏矩阵

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))#[0,506]
            edge_mat_list.append(graph.edge_mat + start_idx[i])#后者为0还是边矩阵
        #cat是concatnate以某一维度进行连接，之前拿中括号包着了现在没包
        Adj_block_idx = torch.cat(edge_mat_list, 1)# [2,5758]
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])#[5758]

        #Add self-loops in the adjacency matrix if learn_eps is False,这里确实是F
        #  i.e., aggregate center nodes and neighbor nodes altogether.
        #如果learn_eps为False，则在邻接矩阵中添加自循环，即将中心节点和相邻节点聚合在一起

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])#[2,506]
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            # 2,6264 加上了自环
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))
        # 这个size 506 506 然后前面的是边加自环6264条
        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        # 为每个图的全部节点建立池化稀疏矩阵
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
        # start_idx:[0,506]
        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling 类型是sum
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
            #扩展成graph（这里是506）个[1, 1, ..., 1]的列表
                elem.extend([1]*len(graph.g))

            # 0,0 到 0,505
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)#一维长度506
        idx = torch.LongTensor(idx).transpose(0,1)# torch.size生成一个元组
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        # 三元组稀疏矩阵第一个是序号对，第二个是内容，第三作为参数给出原有稀疏矩阵的大小
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 
        # 这块儿类型也是sum
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    def cal_sim(self,Adj_block, h):
        i,j =  Adj_block._indices()
        size = Adj_block.size()
        values  = Adj_block._values()
        index = Adj_block._indices()
        values = list(values)
        Adj_block = Adj_block.to_dense()
        for n,(a,b) in enumerate(zip(i,j)): 
            values[n] = torch.cosine_similarity(h[a],h[b],dim=0)
        values = torch.Tensor(values)
        Adj_block = torch.sparse.FloatTensor(index, values, size)
        return Adj_block, h
    def update(self,Adj_block,h,edge_attr):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh, edge_attr)
        e=e*edge_attr
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(edge_attr > 0, e, zero_vec)
        e=F.softmax(e, dim=1)
        #e=torch.exp(e)
        
        #e=DSN(e)
        attention = F.dropout(e, self.dropout, training=self.training)
        
        h_prime=[]
        for i in range(edge_attr.shape[0]):
            h_prime.append(torch.matmul(attention[i],Wh))

        if self.concat:
            h_prime = torch.cat(h_prime,dim=1)
            return F.elu(h_prime),e
        else:
            h_prime = torch.stack(h_prime,dim=0)
            h_prime=torch.sum(h_prime,dim=0)
            return h_prime
    def _prepare_attentional_mechanism_input(self, Wh,edge_attr):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def next_layer(self, h, layer,edge_attr,padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
        # 将相邻节点和中心节点合并在一起    
        # print('h:{}\n layer:{}\n Adj_block:{}\n'.format(h,layer,Adj_block.values))
        if self.neighbor_pooling_type == "max":

            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            #这里是sum 对块和h进行矩阵乘法 506 506 和 506 256
            #这里依据权值每次都对这个adj更新
            #adj的内容是拥有的边和自环 h是节点的向量
            #Adj_block, h = self.cal_sim(Adj_block, h)
            # 这样就不用每次计算相似度了，改为注意力机制下更新的边的权值
                    
            # 在MLPs之前放到GATlayer里！！！！！这样边的权值就是影响节点特征的了，但是还有维度问题
            for att in self.attentions:# 这他妈只有一层，在吓人 把reduce_h的维度也设为输入相同的不好咯
                reduce_h,edge_attr=att(h, edge_attr) # 这个e是注意力系数 这个h其实可以不变的，在这里是用注意力系数和节点属性相乘
                # 要回头确保edgeattr是高维向量，即那个adj映射是不是高维化
            # h,edge_attr = self.update(Adj_block, h,edge_attr) 这个实际上就是layers里的
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling 也就是没用
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        # 相邻节点和中心节点的表示

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
        return h,edge_attr


    def forward(self, batch_graph):
        print('graphCNN')# 一轮迭代一次的情况下 五个一次GIN 四次MLP
        # 其中train一次，test四次，train的话数据是564，
        # test就没准了 1254 1166 557 372
        # 是在test的两个passdata里放进去了二*二次数据

        # torch.cat把张量拼接起来，第0维就是竖着往下堆叠[1,7]->[n,7]
        # print(batch_graph[0].node_features,"gcn\n") gcn一个 
        # 564 = 188 * 3
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        # 那就把向量化的操作放到生成graph的操作里
        edge_attr = torch.cat([graph.edge_features for graph in batch_graph], 0).to(self.device)
        # print('\nX_concat:{}'.format(X_concat.size())) torch.Size([564, 7])
        
        # print('graph_pool:{}'.format(graph_pool.size()))# 两个是前后对应的torch.Size([32, 564])        
        graph_pool = self.__preprocess_graphpool(batch_graph)

        # 每个batch节点数量不一致
        # 维度每次forward是会变化的
        # “7”是标签数目和“32”是批大小是固定的
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            # 这块儿默认也是sum
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)
        # print('\nAdj_block:{}'.format(Adj_block.size())) Adj_block:torch.Size([564, 564])
        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat
        
        # 要保存初始的node_features所以next_layers少了一层 所以是四次mlp
        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            #默认是下列情况 next_layer中有mlps的加入
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                #h本来就是特征的列表因为只有一个图一层包一层的列表 
                # 经过这个操作，h的内容变了形状一个是不变的[506,256]
                h,edge_attr = self.next_layer(h, layer,edge_attr, Adj_block = Adj_block)

            hidden_rep.append(h)

        score_over_layer = 0
        # 感知机内容结束即邻域聚合和更新结束 最后的内容因为包括预测可能有不需要的东西，这里先不管，先考虑前面。
        #perform pooling over all nodes in each graph in every layer
        #读出函数，对每个层中每个图中的所有节点执行池化
        for layer, h in enumerate(hidden_rep):
            # spmm矩阵乘法
            # ([32,564])*([564, 64])=[32,64]
            pooled_h = torch.spmm(graph_pool, h)
            # 线性预测(64,2) 这里是不是把最后的预测给删了就行
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)
        # score_over_layer:torch.Size([32, 2])
        # 就是这个output = model(batch_graph)
        return score_over_layer
