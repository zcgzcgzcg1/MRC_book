import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_Maxpool(nn.Module):
    # word_dim为词向量长度，window_size为CNN窗口长度，output_dim为CNN输出通道数
    def __init__(self, word_dim, window_size, out_channels):
        super(CNN_Maxpool, self).__init__()
        # 1个输入通道，out_channels个输出通道，过滤器大小为window_size x word_dim
        self.cnn = nn.Conv2d(1, out_channels, (window_size, word_dim)) 

    # 输入x为batch组文本，长度seq_len，词向量长度为word_dim, 维度batch x seq_len x word_dim
    # 输出res为所有文本向量，每个向量维度为out_channels
    def forward(self, x):
        # 变成单通道，结果维度batch x 1 x seq_len x word_dim
        x_unsqueeze = x.unsqueeze(1) 
        # CNN, 结果维度batch x out_channels x new_seq_len x 1
        x_cnn = self.cnn(x_unsqueeze) 
        # 删除最后一维，结果维度batch x out_channels x new_seq_len
        x_cnn_result = x_cnn.squeeze(3) 
        # 最大池化，遍历最后一维求最大值，结果维度batch x out_channels
        res, _ = x_cnn_result.max(2)  
        return res

class NLUNet(nn.Module):
    # word_dim为词向量长度，window_size为CNN窗口长度，out_channels为CNN输出通道数，K为类别个数
    def __init__(self, word_dim, window_size, out_channels, K):
        super(NLUNet, self).__init__()
        # CNN和最大池化
        self.cnn_maxpool = CNN_Maxpool(word_dim, window_size, out_channels)  
        # 输出层为全连接层
        self.linear = nn.Linear(out_channels, K)     
    
    # x：输入tensor,维度为batch x seq_len x word_dim
    # 输出class_score,维度是batch x K
    def forward(self, x):
        # 文本向量，结果维度是batch x out_channels
        doc_embed = self.cnn_maxpool(x)  
        # 分类分数，结果维度是batch x K
        class_score = self.linear(doc_embed)     
        return class_score

K = 3     # 三分类
net = NLUNet(10, 3, 15, K)
# 共30个序列，每个序列长度5，词向量维度是10
x = torch.randn(30, 5, 10, requires_grad=True)   
# 30个真值分类，类别为0~K-1的整数 
y = torch.LongTensor(30).random_(0, K) 
optimizer = optim.SGD(net.parameters(), lr=1)  
# res大小为batch x K
res = net(x)
# PyTorch自带交叉熵函数，包含计算softmax
loss_func = nn.CrossEntropyLoss() 
loss = loss_func(res, y)
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
res = net(x)
loss = loss_func(res, y)
print('loss2 =', loss) # loss2应该比loss1小

