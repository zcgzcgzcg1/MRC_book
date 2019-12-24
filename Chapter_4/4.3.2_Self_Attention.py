import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    # dim为向量维度，hidden_dim为自注意力计算的隐藏层维度
    def __init__(self, dim, hidden_dim):
        super(SelfAttention, self).__init__()
        # 参数矩阵W
        self.W = nn.Linear(dim, hidden_dim) 
    
    ''' 
      x: 进行自注意力计算的向量组，batch x n x dim
    '''
    def forward(self, x):
        # 计算隐藏层，结果维度为batch x n x hidden_dim
        hidden = self.W(x)
        # 注意力分数scores，维度为batch x n x n
        scores = hidden.bmm(hidden.transpose(1, 2))
        # 对最后一维进行softmax
        alpha = F.softmax(scores, dim=-1)
        # 注意力向量，结果维度为batch x n x dim
        attended = alpha.bmm(x) 
        return attended

batch = 10
n = 15
dim = 40
hidden_dim = 20
x = torch.randn(batch, n, dim)
self_attention = SelfAttention(dim, hidden_dim)
res = self_attention(x)
print(res.shape) # torch.Size([10, 15, 40])