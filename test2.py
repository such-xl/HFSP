import torch

# 生成模拟数据
predictions = torch.randn(8, 1, 10)  # 假设的网络输出
mask = torch.randint(0, 2, (8, 10), dtype=torch.bool)  # 随机生成mask，其中1代表True即忽略

# 扩展predictions的形状以便与mask对应：[64, 1, 30] => [64, 30]
predictions = predictions.squeeze(1)  # 删除中间维度，形状变为[64, 30]

# 使用mask
min_value = torch.finfo(predictions.dtype).min
masked_predictions = torch.where(mask, torch.tensor(min_value).expand_as(predictions), predictions)
print(predictions)
print(mask)
# 找出每个batch中非忽略值的最大值
max_values, _ = torch.max(masked_predictions, dim=1)  # 返回最大值及其索引，我们只需要值

print(max_values)