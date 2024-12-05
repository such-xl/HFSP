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
    


# class RLNet(nn.Module):
#     def __init__(self, num_machines, num_operations, channels=1):
#         """
#         初始化 CNN 网络
#         :param num_machines: 机器数量（行数）
#         :param num_operations: 工序数量（列数）
#         :param channels: 输入通道数（默认为 1）
#         """
#         super(RLNet, self).__init__()
#         self.num_machines = num_machines
#         self.num_operations = num_operations
#         # 卷积层
#         self.conv1 = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
#         # 池化层
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         # 全连接层
#         flattened_size = (num_machines // 2) * (num_operations // 2) * 64
#         self.fc_shared = nn.Linear(flattened_size, 128)
        
#         # 策略分支
#         self.policy_fc = nn.Linear(128, num_operations * num_machines)
        
#         # 价值分支
#         self.value_fc = nn.Linear(128, 1)

#     def forward(self, x):
#         """
#         前向传播
#         :param x: 输入张量，形状为 (batch_size, channels, num_machines, num_operations)
#         :return: 策略分布和状态价值
#         """
#         # 特征提取部分
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, start_dim=1)  # 展平
        
#         # 共享特征部分
#         shared = F.relu(self.fc_shared(x))
        
#         # 策略输出
#         policy_logits = self.policy_fc(shared)
#         policy = F.softmax(policy_logits.view(-1, self.num_operations, self.num_machines), dim=-1)
        
#         # 价值输出
#         value = self.value_fc(shared)
        
#         return policy, value