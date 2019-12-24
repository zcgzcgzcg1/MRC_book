# 下载GitHub上CoVe的源代码
# git clone https://github.com/salesforce/cove.git
# cd cove
# 安装所需软件包
# pip install -r requirements.txt
# 安装CoVe
# python setup.py develop

# Python代码
import torch
from torchtext.vocab import GloVe
from cove import MTLSTM
# GloVe词表，维度为2.1M×300
glove = GloVe(name='840B', dim=300, cache='.embeddings')
# 输入共2个句子，每个句子中每个单词在词表中的编号
inputs = torch.LongTensor([[10, 2, 3, 0], [7, 8, 10, 3]])
# 2个句子的长度分别为3和4
lengths = torch.LongTensor([3, 4])
# CoVe类
cove = MTLSTM(n_vocab=glove.vectors.shape[0], vectors=glove.vectors, model_cache='.embeddings')
# 每个句子每个单词的CoVe编码，维度为2×4×600
outputs = cove(inputs, lengths)
print(outputs.shape) # torch.Size([2, 4, 600])
