import torch
import torch.nn as nn

layer = nn.Dropout(0.1)  # Dropout层，置零概率为0.1
input = torch.randn(5, 2)
print(input)
output = layer(input)        # 维度仍为5 x 2，每个元素有10%概率为0
print(output)