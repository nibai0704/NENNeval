import torch
import torch.nn as nn
import torch.nn.functional as F

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            2 num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            7 input_dim: dimensionality of input features
            64 hidden_dim: dimensionality of hidden units at ALL layers
            64 output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model一层是文章里做消融实验用的，这里一般都是多层
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            # nn.Linear用来设置全连接层对传入数据应用线性转换，用于创建前馈网络
            # 第一个参数是输入第二个是输出，20, 30 经过128, 20 变成128, 30
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                # torch.nn.BatchNorm1d nn.BatchNorm1d 用于将数据标准化为 0 均值和单位方差。
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    # forward函数里的x实际上是加载的数据data，只要使用了模型就默认加载了
    def forward(self, x):
        #print('mlp\n') #mlp 4个
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            # 这里的层数是mlp的两层
            for layer in range(self.num_layers - 1):
                # print('linear') 4个 也就是只有一层
                # 在最后一层之前，先规范化数据再全连接，再激活
                # 506x8 and 256x64乘不了，那就是里面数据维度的设置有问题
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            #最后一层再全连接输出
            return self.linears[self.num_layers - 1](h)