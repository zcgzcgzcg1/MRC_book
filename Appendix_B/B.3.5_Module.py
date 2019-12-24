import torch
import torch.nn as nn

# 自定制网络为一个class，继承nn.Module
class FirstNet(nn.Module):  
       # 构造函数，输入元素的维度input_dim，全连接后的维度rnn_dim，RNN状态的维度state_dim
    def __init__(self, input_dim, rnn_dim, state_dim):
        super(FirstNet, self).__init__()    # 调用父类nn.Module.__init__()
        # 全连接层，输入维度input_dim，输出维度rnn_dim
        self.linear = nn.Linear(input_dim, rnn_dim)
        # Dropout层，置零概率为0.3
        self.dropout = nn.Dropout(0.3)
        #单层单向GRU，输入维度rnn_dim，状态维度state_dim
        self.rnn = nn.GRU(rnn_dim, state_dim, batch_first=True)  

    # 前向计算函数，x大小为batch x seq_len x input_dim，为长度是seq_len的输入序列
    def forward(self, x):
        # 对全连接层的输出进行dropout，结果维度为batch x seq_len x rnn_dim
        rnn_input = self.dropout(self.linear(x))
        # GRU的最后一个状态，大小为1 x batch x state_dim
        _, hn = self.rnn(rnn_input) 
        # 交换第0、1维，输出维度为batch x 1 x state_dim
        return hn.transpose(0, 1)

net = FirstNet(10, 20, 15)# 获取网络实例
# batch是第0维，共3个序列，每个序列长度5，维度是10
x = torch.randn(3, 5, 10)    
res = net(x)              # res大小为3 x 1 x 15
print(res.shape)          # torch.Size([3, 1, 15])