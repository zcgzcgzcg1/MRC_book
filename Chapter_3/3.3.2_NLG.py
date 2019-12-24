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


class NLGNet(nn.Module):
    # word_dim为词向量长度，window_size为CNN窗口长度，rnn_dim为RNN的状态维度，vocab_size为词汇表大小
    def __init__(self, word_dim, window_size, rnn_dim, vocab_size):
        super(NLGNet, self).__init__()
        # 单词编号与词向量对应参数矩阵
        self.embed = nn.Embedding(vocab_size, word_dim)  
        # CNN和最大池化        
        self.cnn_maxpool = CNN_Maxpool(word_dim, window_size, rnn_dim)
        # 单层单向GRU，batch是第0维
        self.rnn = nn.GRU(word_dim, rnn_dim, batch_first=True) 
        # 输出层为全连接层，产生一个位置每个单词的得分
        self.linear = nn.Linear(rnn_dim, vocab_size)     
    
    # x_id：输入文本的词编号,维度为batch x x_seq_len
    # y_id：真值输出文本的词编号,维度为batch x y_seq_len
    # 输出预测的每个位置每个单词的得分word_scores，维度是batch x y_seq_len x vocab_size
    def forward(self, x_id, y_id):
        # 得到输入文本的词向量，维度为batch x x_seq_len x word_dim
        x = self.embed(x_id) 
        # 得到真值输出文本的词向量，维度为batch x y_seq_len x word_dim
        y = self.embed(y_id) 
        # 输入文本向量，结果维度是batch x cnn_channels
        doc_embed = self.cnn_maxpool(x)
        # 输入文本向量作为RNN的初始状态，结果维度是1 x batch x y_seq_len x rnn_dim
        h0 = doc_embed.unsqueeze(0)
        # RNN后得到每个位置的状态，结果维度是batch x y_seq_len x rnn_dim
        rnn_output, _ = self.rnn(y, h0)
        # 每一个位置所有单词的分数，结果维度是batch x y_seq_len x vocab_size
        word_scores = self.linear(rnn_output)   
        return word_scores

vocab_size = 100                        # 100个单词
net = NLGNet(10, 3, 15, vocab_size)     # 设定网络 
# 共30个输入文本的词id，每个文本长度10
x_id = torch.LongTensor(30, 10).random_(0, vocab_size) 
# 共30个真值输出文本的词id，每个文本长度8
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=1) 
# 每个位置词表中每个单词的得分word_scores，维度为30 x 8 x vocab_size
word_scores = net(x_id, y_id)
# PyTorch自带交叉熵函数，包含计算softmax
loss_func = nn.CrossEntropyLoss()
# 将word_scores变为二维数组，y_id变为一维数组，计算损失函数值
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step() 
word_scores = net(x_id, y_id)
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss2 =', loss) # loss2应该比loss1小
