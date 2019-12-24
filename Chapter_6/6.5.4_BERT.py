# 安装包含BERT在内的Transformer软件包
# pip install pytorch-transformers
# Python代码
import torch
from pytorch_transformers import *
# 使用BERT-base模型，不区分大小写
config = BertConfig.from_pretrained('bert-base-uncased')
# BERT使用的分词工具
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 载入为区间答案型阅读理解任务预设的模型，包括前向网络输出层
model = BertForQuestionAnswering(config)
# 处理训练数据
# 获得文本分词后的单词编号，维度为batch_size(1)×seq_length(4)
input_ids = torch.tensor(tokenizer.encode("This is an example")).unsqueeze(0) 
# 标准答案在文本中的起始和终止位置，维度为batch_size
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
# 获得模型的输出结果
outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
# 得到交叉熵损失函数值loss，以及模型预测答案在每个位置开始和结束的打分start_scores与end_scores，维度均为batch_size(1)×seq_length
loss, start_scores, end_scores = outputs
print('Loss =', loss)
print('Start scores:', start_scores)
print('End scores:', end_scores)
