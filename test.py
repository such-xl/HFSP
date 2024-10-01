import torch
import torch.nn as nn
import torch.nn.functional as F
from scheduling_env.agents import Actor
import numpy as np
# 设置参数
embed_dim =4  # 嵌入维度
num_heads = 2   # 多头注意力中的头数
batch_size = 2  # 批量大小
max_len = 5     # 最大序列长度
actor = Actor(embed_dim,embed_dim,embed_dim,embed_dim,embed_dim,1)
# 创建随机数据
data = torch.zeros(batch_size, max_len, embed_dim)
q = torch.randn(batch_size,max_len, embed_dim)
# 假定第一个批次的最后两个是填充，第二个批次的最后一个是填充
mask = np.ones((batch_size,max_len))
all_true_mask = np.all(mask, axis=1)
mask[all_true_mask, :] = 0
mask = torch.tensor(mask,dtype=bool)
print(mask)
output,weight = actor(data,data,data,mask,mask)
print(output)
print(weight)