import torch
import torch.nn as nn

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
class D3QN(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim,64),
            nn.LeakyReLU(),
            nn.Linear(64,128),
            nn.LeakyReLU(),
            nn.Linear(128,64)
        )

        self.A_net = nn.Linear(64,action_dim)
        self.V_net = nn.Linear(64,1)
    def forward(self,state):
        state_embed = self.linear(state)
        # A Net
        A = self.A_net(state_embed)
        # V Net
        V = self.V_net(state_embed)

        Q = V + A - A.mean(dim=1,keepdim=True)
        return Q

class PolicyNet(nn.Module):
    def __init__(self,state_dim,action_dim,num_heads):
        super().__init__()
        self.attention_0 = nn.MultiheadAttention(embed_dim=state_dim,num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(state_dim,2*state_dim),
            nn.LeakyReLU(),
            nn.Linear(2*state_dim,state_dim),
        )
        self.attention_1 = nn.MultiheadAttention(embed_dim=state_dim,num_heads=num_heads)
        self.feed_forward_1 = nn.Sequential(
            nn.Linear(state_dim,2*state_dim),
            nn.LeakyReLU(),
            nn.Linear(2*state_dim,state_dim),
        )
        self.norm_0 = nn.LayerNorm(state_dim)
        self.norm_1 = nn.LayerNorm(state_dim)
        self.output_linear = nn.Linear(state_dim,action_dim)
    def forward(self,state_0,state_1,state_mask):
        state_embedding, _ = self.attention_0(state_0,state_0,state_0,attn_mask=state_mask)
        state_embedding = self.feed_forward(state_embedding)
        state_embedding = self.norm_0(state_embedding)
        state_embedding, _ = self.attention_1(state_1,state_embedding,state_embedding)
        state_embedding = self.feed_forward_1(state_embedding)
        state_embedding = self.norm_1(state_embedding)
        logits = self.output_linear(state_embedding)
        logits = self.j_m_linear(output)
        # logits = logits + (action_mask-1) * 1e9
        probs = nn.functional.softmax(logits,dim=-1)
        return probs
class QNet(nn.Module):
    def __init__(self,state_dim,machine_dim,action_dim):
        super().__init__()
        self.Q_linear = nn.Sequential(
            nn.Linear(15*state_dim+10,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,1)
        )
    def forward(self,state,actions):
        state = state.view(state.size(0),-1)
        actions = actions.view(actions.size(0),-1)
        output = torch.cat([state,actions],dim=-1)
        Q = self.Q_linear(output)
        return Q


