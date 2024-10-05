import torch

a = torch.tensor([
    [1,2,4],
    [4,7,78],
    [9,9,10]
])
print(a.shape)

b,_ = torch.max(a,dim=1)
b = b.view(-1,1)
print(b.shape)
print(b)

c,_ = torch.max(a,dim=0)
print(c.shape)
print(c)