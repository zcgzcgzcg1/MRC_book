import torch
import torch.nn as nn
# 四层神经网络，输入层大小为30，两个隐藏层大小为50和70，输出层大小为1
linear1 = nn.Linear(30, 50)
linear2 = nn.Linear(50, 70)
linear3 = nn.Linear(70, 1)
# 10组输入数据作为一批次(batch)，每一个输入为30维
x = torch.randn(10, 30)
# 10组输出数据，每一个输出为1维
res = linear3(linear2(linear1(x)))   
print(res)