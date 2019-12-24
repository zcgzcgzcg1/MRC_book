import torch
import torch.nn as nn
import torch.nn.functional as F

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


batch = 10
seq_len = 20
word_dim = 50
window_size = 3
out_channels = 100
x = torch.randn(batch, seq_len, word_dim)
cnn_maxpool = CNN_Maxpool(word_dim, window_size, out_channels)
res = cnn_maxpool(x)
print(res.shape) # torch.Size([10, 100])