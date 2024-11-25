import torch

# 假设 q_values 和 current_mask 是给定的张量
# 例如：
q_values = torch.randn(5, 3)  # 示例 q_values
current_mask = torch.tensor([[1, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)  # 示例 current_mask

# 检查每一行是否有至少一个 True
mask_any = current_mask.any(dim=-1, keepdim=True)

# 创建一个形状匹配的全零张量
default_values = torch.zeros(q_values.size(0), 1).to(q_values.device)

# 使用 where 选择性地从 q_values 中提取值或使用默认值
selected_values = torch.where(
    mask_any,
    q_values.masked_select(current_mask).view(-1, 1),
    default_values
)

print(selected_values)
