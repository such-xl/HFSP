import numpy as np
import torch

mask = torch.tensor([
    [True,True,False],
    [True,False,False],
    [True,True,True],
    [False,False,False]
    ]
)
A = torch.tensor([
    [1],[2],[3],[4]
])
B = torch.tensor([
    [],[],[6],[7]
])
print(B)
print(A[mask[:,0:1]])