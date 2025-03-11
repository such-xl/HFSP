import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,max_seq_len,model_dim):
        # 生成固定的正余弦位置编码
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_seq_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
class Encoder(nn.Module):
    def __init__(self,state_dim,embed_dim,num_heads,dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding()
        self.attn = nn.MultiheadAttention(state_dim,num_heads,dropout=dropout) 
        self.feed_forward = nn.Sequential(
            nn.Linear(state_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim,embed_dim)
        )
        self.norm0 = nn.LayerNorm(state_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,state,key_padding_mask):
        """
            state: (batch_size,seq_len,embed_dim)
            key_padding_mask: (batch_size,seq_len)
        """
        # self attention
        state = state + self.positional_encoding(state.size(1),state.size(2)).to(state.device)
        state = state.transpose(0,1)
        attn_output, _ = self.attn(state,state,state,key_padding_mask=key_padding_mask)
        # post layer norm
        attn_output = attn_output.transpose(0,1)
        attn_output = self.norm0(attn_output+self.dropout(attn_output))

        # feed forward
        ff_output = self.feed_forward(attn_output)
        # post layer norm
        #output = self.norm1(attn_output+self.dropout(ff_output))
        output = self.norm1(self.dropout(ff_output))
        return output

class Decoder(nn.Module):
    def __init__(self,x_dim,num_heads,dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding()
        self.attn_0 = nn.MultiheadAttention(x_dim,num_heads,dropout=dropout)
        self.attn_1 = nn.MultiheadAttention(x_dim,num_heads,dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(x_dim, 2*x_dim),
            nn.GELU(),
            nn.Linear(2*x_dim, x_dim)
        )

        self.norm0 = nn.LayerNorm(x_dim)
        self.norm1 = nn.LayerNorm(x_dim)
        self.norm2 = nn.LayerNorm(x_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,tgt,mem,tgt_padding_mask):
        # self attention
        tgt = tgt + self.positional_encoding(tgt.size(1),tgt.size(2)).to(tgt.device)
        tgt = tgt.transpose(0,1)
        mem = mem.transpose(0,1)
        attn_output, _ = self.attn_0(tgt,tgt,tgt,key_padding_mask=tgt_padding_mask)
        attn_output = attn_output.transpose(0,1)
        # post layer norm
        attn_output  = self.norm0(attn_output + self.dropout(attn_output))

        # cross attention
        attn_output = attn_output.transpose(0,1)
        attn_output, _ = self.attn_1(attn_output,mem,mem)
        attn_output = attn_output.transpose(0,1)
        # post layer norm
        attn_output = self.norm1(attn_output+self.dropout(attn_output))

        # feed forward
        ff_output = self.feed_forward(attn_output)
        # post layer norm
        output = self.norm2(attn_output+self.dropout(ff_output))
        return output

class D3QN(nn.Module):
    def __init__(self,state_dim,x_dim,action_dim,num_heads,dropout):
        super().__init__()
        self.encoder = Encoder(state_dim,x_dim,num_heads,dropout)
        self.decoder = Decoder(x_dim,num_heads,dropout)
        self.A_net = nn.Linear(x_dim,action_dim)
        self.V_net = nn.Linear(x_dim,1)
        self.gule = nn.GELU()
    def forward(self,mem,machine_action,padding_mask):
        output = self.decoder(machine_action,mem,padding_mask)[:,-1,:]
        # A Net
        A = self.gule(self.A_net(output))
        # V Net
        V = self.gule(self.V_net(output))
        Q = V + A - A.mean(dim=-1,keepdim=True)
        return Q


import torch
import torch.nn as nn
import torch.optim as optim

# 共享表征层（多任务共享特征提取）
class SharedRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_dim):
        super(SharedRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, shared_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x  # 输出通用表征

# 多任务模型
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_dim, num_machines):
        super(MultiTaskModel, self).__init__()
        
        # 共享表征层
        self.shared_representation = SharedRepresentation(input_dim, hidden_dim, shared_dim)
        
        # 任务特定层
        self.makespan_layer = nn.Linear(shared_dim, num_machines)  # 最小化加工时间
        self.balance_layer = nn.Linear(shared_dim, num_machines)   # 均衡机器负载
        self.urgent_layer = nn.Linear(shared_dim, num_machines)    # 处理紧急任务

    def forward(self, state):
        shared_features = self.shared_representation(state)
        makespan_output = self.makespan_layer(shared_features)  # 选择机器
        balance_output = self.balance_layer(shared_features)    # 选择机器
        urgent_output = self.urgent_layer(shared_features)      # 选择机器
        return makespan_output, balance_output, urgent_output