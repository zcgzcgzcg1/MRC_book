# 安装allennlp软件包
# pip install allennlp / pip3 install allennlp
# Python代码 （必须使用Python 3.6版本）
import torch
from torch import nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.nn.util import remove_sentence_boundaries
# 预训练模型下载地址
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# 获得ELMo编码器的类
elmo_bilm = ElmoEmbedder(options_file, weight_file).elmo_bilm
elmo_bilm.cuda()
sentences = [['Today', 'is', 'sunny', '.'], ['Hello', '!']]
# 获得所有单词的字符id，维度batch_size(2)×max_sentence_len(4)×word_len(50)
character_ids = batch_to_ids(sentences).cuda()
# 获得ELMo输出
bilm_output = elmo_bilm(character_ids)
# ELMo编码
layer_activations = bilm_output['activations']
# 每个位置是否有单词
mask_with_bos_eos = bilm_output['mask']
# 去掉ELMo加上的句子开始和结束标识符
without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos) for layer in layer_activations]
# 获得三层ELMo编码，每层1024维，维度3×batch_size(2)×max_sentence_len(4)×1024
all_layers = torch.cat([ele[0].unsqueeze(0) for ele in without_bos_eos], dim=0)
# 求加权和时每层的权重参数
s = nn.Parameter(torch.Tensor([1., 1., 1.]), requires_grad=True).cuda()
# 权重和为1
s = F.softmax(s, dim=0)
# 求加权和时的相乘因子γ
gamma = nn.Parameter(torch.Tensor(1, 1), requires_grad=True).cuda()
# 获得ELMo编码，维度为batch_size(2)×max_sentence_len(4)×1024
res = (all_layers[0]*s[0]+ all_layers[1]*s[1]+ all_layers[2]*s[2]) * gamma
print(res.shape)
