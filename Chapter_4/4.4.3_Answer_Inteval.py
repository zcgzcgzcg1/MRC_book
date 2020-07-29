import torch
import numpy as np
import torch.nn.functional as F

# 设文本共m个词，prob_s是大小为m的开始位置概率，prob_e是大小为m的结束位置概率，均为一维PyTorch张量
# L为答案区间可以包含的最大的单词数
# 输出为概率最高的区间在文本中的开始和结束位置
def get_best_interval(prob_s, prob_e, L):
    # 获得m×m的矩阵，其中prob[i,j]=prob_s[i]×prob_e[j]
    prob = torch.ger(prob_s, prob_e) 
    # 将prob限定为上三角矩阵，且只保留主对角线及其右上方L-1条对角线的值，其他值清零
    # 即如果i>j或j-i+1>L，设置prob[i, j] = 0
    prob.triu_().tril_(L - 1) 
    # 转化成为numpy数组
    prob = prob.numpy()
    # 获得概率最高的答案区间，开始位置为第best_start个词, 结束位置为第best_end个词
    best_start, best_end = np.unravel_index(np.argmax(prob), prob.shape)
    return best_start, best_end

sent_len = 20
L = 5
prob_s = F.softmax(torch.randn(sent_len), dim=0)
prob_e = F.softmax(torch.randn(sent_len), dim=0)
best_start, best_end = get_best_interval(prob_s, prob_e, L)
print(best_start, best_end)
