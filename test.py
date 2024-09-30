import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置参数
embed_dim =4  # 嵌入维度
num_heads = 2   # 多头注意力中的头数
batch_size = 2  # 批量大小
max_len = 5     # 最大序列长度
# 创建随机数据
data = torch.randn(batch_size, max_len, embed_dim)
q = torch.randn(batch_size,10, embed_dim)
# 假定第一个批次的最后两个是填充，第二个批次的最后一个是填充
mask = torch.tensor([
    [False, False, False, True, True],  # 第一个序列的mask，True代表这个位置是填充
    [False, False, False, False, True]  # 第二个序列的mask
])
print('data',data.shape)
print(data)

# 由于mask需要在attention中填充应用，所以需要进行扩展
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
fc = nn.Linear(embed_dim,1)
#print(key_padding_mask.shape)
# 转化data为适合多头注意力输入的形式
key = value = data.transpose(0, 1)  # 注意torch.nn.MultiheadAttention需要 (seq_len, batch, embed_dim)
query = q.data.transpose(0,1)
# 应用多头注意力
attn_output, attn_output_weights = multihead_attn(query, key, value,key_padding_mask = mask)
attn_output = attn_output.transpose(0,1)

softmax = nn.Softmax(dim=2)
print(softmax(attn_output))
# attn_output = attn_output[-1,:,:].unsqueeze(0)
# print(attn_output.transpose(0,1))

# result = torch.cat((query,attn_output),dim=0)
# result=result.transpose(0,1)

print([0].extend([1,12]))
import random

print(random.randint(-1,1))