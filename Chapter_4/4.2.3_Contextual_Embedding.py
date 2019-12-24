import torch
import torch.nn as nn

class Contextual_Embedding(nn.Module):
    # word_dim为词向量维度，state_dim为RNN状态维度，rnn_layer为RNN层数
    def __init__(self, word_dim, state_dim, rnn_layer):
        super(Contextual_Embedding, self).__init__()
        #多层双向GRU，输入维度word_dim，状态维度state_dim
        self.rnn = nn.GRU(word_dim, state_dim, num_layers=rnn_layer, bidirectional=True, batch_first=True)  

    # 输入x为batch组文本，每个文本长度seq_len，每个词用一个word_dim维向量表示，输入维度为batch x seq_len x word_dim
    # 输出res为所有单词的上下文向量表示，维度是batch x seq_len x out_dim
    def forward(self, x):
        # 结果维度batch x seq_len x out_dim，其中out_dim=2 x state_dim，包括两个方向
        res, _ = self.rnn(x) 
        return res

batch = 10
seq_len = 20
word_dim = 50
state_dim = 100
rnn_layer = 2
x = torch.randn(batch, seq_len, word_dim)
context_embed = Contextual_Embedding(word_dim, state_dim, rnn_layer)
res = context_embed(x)
print(res.shape) # torch.Size([10, 20, 200])