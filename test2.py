import torch
import torch.nn as nn

# 输入张量: [batch_size, seq_len]
x = torch.randn(2, 10)  # 单样本，10 个时间步长或特征

# 创建 1D 卷积层
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=5)

# 调整输入形状为 [batch_size, in_channels, seq_len]
x = x.unsqueeze(1)  # 增加一维使得格式为 [batch_size, in_channels, seq_len]
print(x.shape)
# 应用卷积
y = conv(x)

print(y.shape)  # 输出新张量的形状
print(y)
