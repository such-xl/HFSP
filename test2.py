import torch

actions = torch.zeros([10,5])
for i in range(10):
    action = torch.randint(i+1,i+2,[10]).unsqueeze(-1)
    actions[:,i] = action
    print(actions)