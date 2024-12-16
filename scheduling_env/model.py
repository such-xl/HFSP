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
    def __init__(self,state_dim,machine_dim,action_dim):
        super().__init__()
        self.machine_dim = machine_dim
        self.job_linear = nn.Sequential(
            nn.Linear(state_dim,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.machine_linear = nn.Sequential(
            nn.Linear(machine_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            nn.LeakyReLU(),
        )
        self.j_m_linear = nn.Sequential(
            nn.Linear(11,32),
            nn.LeakyReLU(),
            nn.Linear(32,64),
            nn.LeakyReLU(),
        )
        self.A_net = nn.Linear(64,action_dim)
        self.V_net = nn.Linear(64,1)
    def forward(self,state,action_mask):
        job_state = state[:,:-1,:]
        # job_state = job_state.view(job_state.size(0),-1)
        machine_state = state[:,-1,:self.machine_dim]
        job_embedding = self.job_linear(job_state).squeeze(1)
        machine_embedding = self.machine_linear(machine_state)
        output = torch.cat([job_embedding,machine_embedding],dim=-1)
        output = self.j_m_linear(output)
        # A Net
        A = self.A_net(output)
        # V Net
        V = self.V_net(output)
        # 计算有效动作的均值
        valid_A_mean = (A * action_mask).sum(dim=1, keepdim=True) / action_mask.sum(dim=1, keepdim=True)

        # 对有效动作归一化
        normalized_A = A - valid_A_mean

        Q = V + normalized_A
        return Q

class PolicyNet(nn.modules):
    def __init__(self,state_dim,machine_dim,action_dim):
        super().__init__()
        self.machine_dim = machine_dim
        self.job_linear = nn.Sequential(
            nn.Linear(state_dim,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.machine_linear = nn.Sequential(
            nn.Linear(machine_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            nn.LeakyReLU(),
        )
        self.j_m_linear = nn.Sequential(
            nn.Linear(11,32),
            nn.LeakyReLU(),
            nn.Linear(32,action_dim),
            nn.LeakyReLU(),
        )
    def forward(self,state,action_mask):
        job_state = state[:,:-1,:]
        machine_state = state[:,-1,:self.machine_dim]
        job_embedding = self.job_linear(job_state).squeeze(1)
        machine_embedding = self.machine_linear(machine_state)
        output = torch.cat([job_embedding,machine_embedding],dim=-1)
        logits = self.j_m_linear(output)
        logits = logits * (action_mask-1) * 1e9
        probs = nn.F.softmax(logits,dim=-1)
        return probs
class PolicyNet(nn.modules):
    def __init__(self,state_dim,machine_dim,action_dim):
        super().__init__()
        self.machine_dim = machine_dim
        self.job_linear = nn.Sequential(
            nn.Linear(state_dim,32),
            nn.LeakyReLU(),
            nn.Linear(32,1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.machine_linear = nn.Sequential(
            nn.Linear(machine_dim,16),
            nn.LeakyReLU(),
            nn.Linear(16,1),
            nn.LeakyReLU(),
        )
        self.j_m_linear = nn.Sequential(
            nn.Linear(11,32),
            nn.LeakyReLU(),
            nn.Linear(32,action_dim),
            nn.LeakyReLU(),
        )
    def forward(self,state,action_mask):
        job_state = state[:,:-1,:]
        machine_state = state[:,-1,:self.machine_dim]
        job_embedding = self.job_linear(job_state).squeeze(1)
        machine_embedding = self.machine_linear(machine_state)
        output = torch.cat([job_embedding,machine_embedding],dim=-1)
        logits = self.j_m_linear(output)
        logits = logits * (action_mask-1) * 1e9
        probs = nn.F.softmax(logits,dim=-1)
        return probs


