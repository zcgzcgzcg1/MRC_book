import torch
import torch.nn as nn

# 双层GRU输入元素维度是10，状态维度是20，batch是第1维
rnn = nn.GRU(10, 20, num_layers=2)    
# 一批次共3个序列，每个序列长度5，维度是10，注意batch是第1维
x = torch.randn(5, 3, 10) 
# 初始状态，共3个序列，2层，维度是20
h0 = torch.randn(2, 3, 20)
# output是所有的RNN状态，大小为5 x 3 x 20；hn大小为2 x 3 x 20，为RNN最后一个状态
output, hn = rnn(x, h0) 

print(output.shape) # torch.Size([5, 3, 20])
print(hn.shape) # torch.Size([2, 3, 20])