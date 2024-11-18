import torch
import torch.nn as nn

action_mask = torch.tensor([[True, True, True, True, True]]).to('cuda')

print(action_mask)

action_mask_copy = action_mask.cpu().numpy()
print(action_mask_copy)

b = torch.tensor([1]).to('cuda')
print(b)
a = b.cpu().item()
print(a)
print(type(a))