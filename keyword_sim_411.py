# 这个是专为GAT设计的数据格式，可能也是未来要使用的
# 邻接矩阵：边的，节点的，节点和边的
# 特征矩阵，节点的，边的
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from operator import eq
import numpy as np
from models.utils import DSN
import networkx as nx

# 这里由create_ds.py生成的数据集进行向量化并计算关键词相似度以此作为连边
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")
model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection",
                                                            output_hidden_states = True,
    )
dataset_path = 'C:/Users/you/Desktop/paper_rec/data/nenn_dataset.txt'


def tokenize(input):
    # 输出的分词文本 1用在bert向量生成上 2用在从句子中摘对应词上
    marked_text = "[CLS] " + input + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    return tokenized_text,indexed_tokens
def vectorize(tokenized_text,indexed_tokens):
    # 返回sum之后的向量 应该效果不好 待定吧
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[1]# 隐藏状态 长度5 每个 1 6 256 这个长度是由模型决定的
    token_embeddings = torch.stack(hidden_states, dim=0) # 5 1 6 256
    token_embeddings = torch.squeeze(token_embeddings, dim=1) # 5 6 256
    token_embeddings = token_embeddings.permute(1,0,2) # 6 5 256
    token_vecs_sum = []
    # 这里cat的话会出现维数不一致的情况 以后可以用别的方式
    for token in token_embeddings: # token 5 256
        sum_vec = torch.sum(token[-4:], dim=0)  # cat_vec 256
        token_vecs_sum.append(sum_vec)
    token_vecs_sum = torch.stack(token_vecs_sum,dim=0) # n 256

    return token_vecs_sum

keywords_dict = {}
all_keywords = []
with open(dataset_path, 'r',encoding='utf-8')as f:
    #全部文章的数量
    n_g = int(f.readline().strip())
    for index_g in range(n_g):
        #文章中句子的数量
        all_keywords.append(f.readline().strip().rstrip(';').split('; '))
        line_num = int(f.readline().strip())
        keyword_index_g = 0        
        for index_l in range(line_num):
            line = f.readline().strip().split('; ')
            keywords = []
            # 第0位是关键词数量
            keyword_num = int(line[0])
            for k in range(keyword_num):
                keywords.append(line[1+k])
            # 剩下的最后一个是句子
            sentence = line[1+keyword_num]
            # 生成句级向量和关键词向量
            word_vec = []
            word_token = []
            sent_vec = []
            sent_token = []
            # 处理关键词列表
            for word in keywords:
                tokenized_text,indexed_tokens = tokenize(word)
                token_vecs_sum = vectorize(tokenized_text,indexed_tokens)
                # 这里的大小由token的长度有关
                word_vec.append(token_vecs_sum)
                word_token.append(tokenized_text)
            # 处理句子
            tokenized_text,indexed_tokens = tokenize(sentence)
            token_vecs_sum = vectorize(tokenized_text,indexed_tokens)
            sent_vec = token_vecs_sum
            sent_token = tokenized_text
            # 在句子里摘出关键词向量 排除sep cls
            for i in range(1,len(sent_token)-1):
                for j in range(len(word_token)):
                    if eq(sent_token[i:i+len(word_token[j])-2],word_token[j][1:-1]):
                        # 文章中出现多少次关键词就能摘出来多少
                        tmp1 = torch.sum(sent_vec[i:i+len(word_token[j])-2],dim=0)
                        tmp2 = torch.sum(word_vec[j][1:-1],dim=0)
                        # 怕万一一句里有两个一样关键词 把在关键词列表位置也放上吧X
                        # 这里改成由文章中出现的关键词位次 因为以相似度算了就都相似度吧
                        # 又改了 为了方便矩阵定位 改成在字典的len了
                        #keywords_dict[(index_g,all_keywords[index_g].index(keywords[j]),keywords[j])] = tmp1.size() 
                        keywords_dict[(index_g,len(keywords_dict),keywords[j])] = tmp1
                        keyword_index_g += 1 
                        # 加和操作之后 和源文本的相似度由0.36-0.79
                        # print(torch.cosine_similarity(tmp1,tmp2,dim=0))
# print(keywords_dict)
# 构建纯0矩阵
length_g = len(keywords_dict.keys())
zero_net = np.zeros(length_g*length_g)
zero_net.resize(length_g,length_g)
# 进行相似度计算
for i_key,i_val in keywords_dict.items():
    for j_key,j_val in keywords_dict.items():
        sim = float(torch.cosine_similarity(i_val,j_val,dim=0))
        if sim > 0.7 and sim < 1:
            # print(sim,i_key,j_key)
            zero_net[i_key[1]][j_key[1]] = sim

#print(np.count_nonzero(zero_net))#255025 0.6 32508  0.65 13994 0.7 5954

node_features = []
edge_features = [] # 那节点表示边不好咯
node_adj = np.zeros((len(zero_net),len(zero_net)))
for i,v in enumerate(keywords_dict.values()):    
    # 因为节点是按顺序放的 特征按顺序添加就好
    node_features.append(v)
    for j in range(len(zero_net[0])):
        if(zero_net[i][j] != 0 and j<i):#2879 and j!=i 3075 这里就是对角线有没有东西的上三角矩阵的区别
            # print(zero_net[i][j])
            node_adj[i][j] = 1 #zero_net[i][j]
            edge_features.append(torch.FloatTensor([zero_net[i][j]]))

#print(len(edge_features))# 5758 = 2879*2 下面全降到2879了
row_indices, col_indices = np.nonzero(node_adj) # 两个都是5758
edge_adj = np.zeros((len(row_indices), len(row_indices))) # 5758 5758

node_edge_adj = np.zeros((len(zero_net),len(row_indices)))# 506 5758
for i in range(len(row_indices)):
    for j in range(len(row_indices)):
        if(row_indices[i] == col_indices[j]):
            edge_adj[i][j] =1
            edge_adj[j][i] =1
            node_edge_adj[row_indices[i]][i] = 1
            # edge_features[i] = node_adj[row_indices[i]][col_indices[j]]
for i in range(len(row_indices)):
    for j in range(len(row_indices)):
        if(col_indices[i] == col_indices[j] or row_indices[i] == row_indices[j]):
            edge_adj[i][j] =1
            edge_adj[j][i] =1
            # edge_features[i] = node_adj[row_indices[i]][col_indices[j]]
#print(len(edge_adj))
# print(g.edges().data('weight'))# 每一项就是这个样子的(503, 505, 0.9304949045181274)
if node_features != []:
    node_features = torch.stack(node_features,dim=0)
if edge_features != []:
    edge_features = torch.stack(edge_features,dim=0)
# ([560, 256])


def pass_data():
    return node_features,edge_features,node_adj,edge_adj,node_edge_adj

