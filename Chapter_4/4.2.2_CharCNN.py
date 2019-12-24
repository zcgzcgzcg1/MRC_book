import torch
import torch.nn as nn
import torch.nn.functional as F

class Char_CNN_Maxpool(nn.Module):
    # char_num为字符表大小，char_dim为字符向量长度，window_size为CNN窗口长度，output_dim为CNN输出通道数
    def __init__(self, char_num, char_dim, window_size, out_channels):
        super(Char_CNN_Maxpool, self).__init__()
        # 字符表的向量，共char_num个向量，每个维度为char_dim
        self.char_embed = nn.Embedding(char_num, char_dim)
        # 1个输入通道，out_channels个输出通道，过滤器大小为window_size x char_dim
        self.cnn = nn.Conv2d(1, out_channels, (window_size, char_dim)) 

    # 输入char_ids为batch组文本，每个文本长度seq_len，每个词含word_len个字符编号（0~char_num-1），输入维度为batch x seq_len x word_len
    # 输出res为所有单词的字符向量表示，维度是batch x seq_len x out_channels
    def forward(self, char_ids):
        # 根据字符编号得到字符向量，结果维度batch x seq_len x word_len x char_dim
        x = self.char_embed(char_ids)
        # 合并前两维并变成单通道，结果维度(batch x seq_len) x 1 x word_len x char_dim
        x_unsqueeze = x.view(-1, x.shape[2], x.shape[3]).unsqueeze(1) 
        # CNN, 结果维度(batch x seq_len) x out_channels x new_seq_len x 1
        x_cnn = self.cnn(x_unsqueeze) 
        # 删除最后一维，结果维度(batch x seq_len) x out_channels x new_seq_len
        x_cnn_result = x_cnn.squeeze(3) 
        # 最大池化，遍历最后一维求最大值，结果维度(batch x seq_len) x out_channels
        res, _ = x_cnn_result.max(2)  
        return res.view(x.shape[0], x.shape[1], -1)

batch = 10
seq_len = 20
word_len = 12
char_num = 26
char_dim = 10
window_size = 3
out_channels = 8
char_cnn = Char_CNN_Maxpool(char_num, char_dim, window_size, out_channels)
char_ids = torch.LongTensor(batch, seq_len, word_len).random_(0, char_num - 1)
res = char_cnn(char_ids)
print(res.shape) # torch.Size([10, 20, 8])
