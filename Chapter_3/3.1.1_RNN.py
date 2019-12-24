import torch
import torch.nn as nn

class BiRNN(nn.Module):
    # word_dim为词向量长度，hidden_size为RNN隐状态维度
    def __init__(self, word_dim, hidden_size):
        super(BiRNN, self).__init__()
        # 双向GRU，输入的张量第一维是batch大小
        self.gru = nn.GRU(word_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True) 

    # 输入x为batch组文本，长度seq_len，词向量长度为word_dim, 维度batch x seq_len x word_dim
    # 输出为文本向量，维度为batch x (2 x hidden_size)
    def forward(self, x):
        batch = x.shape[0]
        # output为每个单词对应的最后一层RNN的隐状态，维度为batch x seq_len x (2 x hidden_size)
        # last_hidden为最终的RNN状态，维度为2 x batch x hidden_size
        output, last_hidden = self.gru(x)
        return last_hidden.transpose(0,1).contiguous().view(batch, -1)  

batch = 10
seq_len = 20
word_dim = 50
hidden_size = 100
x = torch.randn(batch, seq_len, word_dim)
birnn = BiRNN(word_dim, hidden_size)
res = birnn(x)
print(res.shape) # torch.Size([10, 200])
