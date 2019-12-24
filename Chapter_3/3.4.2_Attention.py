import torch
import torch.nn as nn
import torch.nn.functional as F

'''
  a: 被注意的的向量组，batch x m x dim 
  x: 进行注意力计算的向量组，batch x n x dim
'''
def attention(a, x):
    # 内积计算注意力分数，结果维度为batch x n x m
    scores = x.bmm(a.transpose(1, 2))
    # 对最后一维进行softmax
    alpha = F.softmax(scores, dim=-1)
    # 注意力向量，结果维度为batch x n x dim
    attended = alpha.bmm(a) 
    return attended

batch = 10
m = 20
n = 30
dim = 15
a = torch.randn(batch, m, dim)
x = torch.randn(batch, n, dim)
res = attention(a, x)
print(res.shape) # torch.Size([10, 30, 15])

