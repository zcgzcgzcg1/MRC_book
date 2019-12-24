import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSum(nn.Module):
    # 输入的词向量维度为word_dim
    def __init__(self, word_dim):
        super(WeightedSum, self).__init__()
        self.b = nn.Linear(word_dim, 1) # 参数张量

    # x：输入tensor,维度为batch x seq_len x word_dim
    # 输出res,维度是batch x word_dim
    def forward(self, x):
        # 内积得分，维度是batch x seq_len x 1
        scores = self.b(x)
        # softmax运算，结果维度是batch x seq_len x 1
        weights = F.softmax(scores, dim = 1) 
        # 用矩阵乘法实现加权和，结果维度是batch x word_dim x 1
        res = torch.bmm(x.transpose(1, 2), weights)  
        # 删除最后一维，结果维度是batch x word_dim 
        res = res.squeeze(2)  
        return res

batch = 10
seq_len = 20
word_dim = 50
x = torch.randn(batch, seq_len, word_dim)
weighted_sum = WeightedSum(word_dim)
res = weighted_sum(x)
print(res.shape) # torch.Size([10, 50])