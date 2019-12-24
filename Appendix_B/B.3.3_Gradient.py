import torch
a = torch.ones(1)         # 一个1维向量，值为1
a = a.cuda()              # 将a放入GPU，如果本机没有GPU，注释此句
print(a.requires_grad)    # False
a.requires_grad = True    # 设定a需要计算导数
b = torch.ones(1)
x = 3 * a + b             # x是最终结果
print(x.requires_grad)    # True，因为a需要计算导数，所以x需要计算导数
x.backward()              # 计算所有参数的导数
print(a.grad)             # tensor([ 3.])，导数为3 
