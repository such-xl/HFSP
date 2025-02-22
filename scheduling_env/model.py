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
    def __init__(self,state_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim,64),
            nn.LeakyReLU(),
            nn.Linear(64,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,2)
        )
    def forward(self,state):
        state_embed = self.linear(state)
        alpha = torch.exp(state_embed[:, 0])  # 取第一个值作为 alpha
        beta = torch.exp(state_embed[:, 1])   # 取第二个值作为 beta
        return alpha, beta
    

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


