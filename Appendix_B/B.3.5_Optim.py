import torch
import torch.nn as nn
import torch.optim as optim   # 优化器软件包

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

net = FirstNet(10, 20, 15)
net.train()              # 将FirstNet置为训练模式（启用Dropout）
#net.cuda()               # 如果有GPU，执行此语句将FirstNet的参数放入GPU 

# 随机定义训练数据
# 共30个序列，每个序列长度5，维度是10
x = torch.randn(30, 5, 10)  
y = torch.randn(30, 1)       # 30个真值
# 随机梯度下降SGD优化器，学习率为0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)  
for batch_id in range(10):
    # 获得当前批次的数据，batch_size=3
    x_now = x[batch_id * 3: (batch_id + 1) * 3]
    y_now = y[batch_id * 3 : (batch_id + 1) * 3]
    res = net(x_now)                         # RNN结果res，维度为3115
    y_hat, _ = torch.max(res, dim=2)      # 最终预测张量y_hat，维度为31
    # 均方差损失函数
    loss = torch.sum(((y_now - y_hat) ** 2.0)) / 3  
    print('loss =', loss)
    optimizer.zero_grad()                   # 对net里所有张量的导数清零
    loss.backward()                          # 自动实现反向传播
    optimizer.step()                         # 按优化器的规则沿导数反方向移动每个参数

net.eval()            # 训练完成后，将FirstNet置为测试模式（Dropout不置零，不删除神经元）
y_pred = net(x)       # 获得测试模式下的输出
