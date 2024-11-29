import torch
def _generate_positional_encoding(max_seq_len, model_dim):
     # 生成固定的正余弦位置编码
     position = torch.arange(max_seq_len).unsqueeze(1)
     div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))
     pe = torch.zeros(max_seq_len, model_dim)
     pe[:, 0::2] = torch.sin(position * div_term)
     pe[:, 1::2] = torch.cos(position * div_term)
     return pe.unsqueeze(0)

pe = _generate_positional_encoding(10, 16)
print(pe.shape)
A = torch.ones(2,10,16)
A = A + pe
print(A)