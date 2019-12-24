import torch
import torch.nn as nn

# 卷积神经网络，输入通道有1个，输出通道3个，过滤器大小为5
conv = nn.Conv2d(1, 3, 5)
# 10组输入数据作为一批次(batch)，每一个输入为单通道32 x 32矩阵
x = torch.randn(10, 1, 32, 32)  
# y维度为10 x 3 x 28 x 28，表示输出10组数据，每一个输出为3通道28 x 28矩阵 (28=32-5+1)
y = conv(x)
print(y.shape) # torch.Size([10, 3, 28, 28]