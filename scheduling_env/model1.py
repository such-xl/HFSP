import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2 * input_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, output_dim),
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=-1)  # Action probability distribution
        return x

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()

        # CNN 提取局部特征
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Transformer 提取全局特征
        self.state_embedding = nn.Linear(64, 128)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=1, dim_feedforward=512, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # 共享 MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 任务特定 MLP
        self.critic_U = nn.Linear(64, 1)  
        self.critic_trad = nn.Linear(64, 1)  

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度 [50, 10, 6] -> [50, 6, 10]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # 恢复维度 [50, 10, 64]

        x = self.state_embedding(x)  
        x = self.transformer_encoder(x)  
        x = x.mean(dim=1)  

        x = self.shared_mlp(x)
        return self.critic_U(x), self.critic_trad(x)

class PPO:
    def __init__(
            self,
            local_state_dim,
            local_state_len,
            global_state_dim,
            global_state_len,
            act_dim,
            lr,
            gamma,
            lmbda,
            eps,
            epochs,
            weights,  # 权重参数
            batch_size,
            device=torch.device("cpu")
            ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs 
        self.batch_size = batch_size
        self.device = device
        self.weights = torch.tensor(weights, dtype=torch.float).to(device)  # 添加权重参数
        self.actor = Actor(local_state_dim*local_state_len, act_dim).to(device=device)
        self.critic = Critic(global_state_dim).to(device=device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = {'local_state': [], 'global_state':[], 'actions': [], 'rewards': [], 'next_local_state': [],'next_global_state':[], 'dones': [], 'mask':[], 'next_mask':[]}

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def store_transition(self, local_state, global_state, action, reward, next_local_state, next_global_state, done, mask, next_mask):
        self.memory['local_state'].append(local_state)
        self.memory['global_state'].append(global_state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_local_state'].append(next_local_state)
        self.memory['next_global_state'].append(next_global_state)
        self.memory['dones'].append(done)
        self.memory['mask'].append(mask)
        self.memory['next_mask'].append(next_mask)
    

    def compute_advantage(self, td_delta):
        advantage = [] 
        adv = 0.0
        for delta in reversed(td_delta):
            adv = delta + self.gamma * self.lmbda * adv
            advantage.insert(0, adv)
        return torch.tensor(advantage, dtype=torch.float).to(self.device)
    
    def update(self):
        local_state = torch.tensor(self.memory['local_state'], dtype=torch.float).to(self.device)
        global_state = torch.tensor(self.memory['global_state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(self.memory['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32, device=self.device)  # 2个目标
        next_local_state = torch.tensor(self.memory['next_local_state'], dtype=torch.float).to(self.device)
        next_global_state = torch.tensor(self.memory['next_global_state'], dtype=torch.float).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.float32, device=self.device)
        mask = torch.tensor(self.memory['mask'], dtype=torch.bool).to(self.device)
        next_mask = torch.tensor(self.memory['next_mask'], dtype=torch.bool).to(self.device)
        # rewards_U = rewards[:, 0]
        # rewards_Trad = rewards[:, 1]

        old_log_probs = torch.log(self.actor(local_state).gather(1, actions)).detach()
        
        with torch.no_grad():
            value_U, value_Trad = self.critic(global_state)
            value_next_U, value_next_Trad = self.critic(next_global_state)
            value_U = value_U.squeeze(-1)
            value_Trad = value_Trad.squeeze(-1)
            value_next_U = value_next_U.squeeze(-1)
            value_next_Trad = value_next_Trad.squeeze(-1)
        
         # 计算 两个目标 TD 目标值
        td_target_U = rewards + self.gamma*value_next_U*(1-dones)
        td_target_Trad = rewards + self.gamma*value_next_Trad*(1-dones)
        advantage = self.compute_advantage(td_target_U - value_U) 
        # advantage_Trad = self.compute_advantage(td_target_Trad - value_Trad) 
        #动态调整权重

        # total_adv = abs(advantage_U.mean()) + abs(advantage_Trad.mean()) +1e-8
        # weight_U = abs(advantage_U.mean()) / total_adv
        # weight_Trad = abs(advantage_Trad.mean()) / total_adv
        # advantage = weight_U * advantage_U + weight_Trad * advantage_Trad
        # advantage =  self.weights[0] * advantage_U + self.weights[1] * advantage_Trad 
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)

        actor_loss_epochs = 0
        loss_U_epochs = 0
        loss_trad_epochs = 0
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(local_state).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            value_U, value_Trad = self.critic(global_state)
            value_U = value_U.squeeze(-1)
            # value_Trad = value_Trad.squeeze(-1)
            
            loss_fn = nn.MSELoss()
            loss_U = loss_fn(value_U, td_target_U.detach())
            # loss_Trad = loss_fn(value_Trad, td_target_Trad.detach())
            

            actor_loss_epochs += actor_loss.item()
            loss_U_epochs += loss_U.item()
            # loss_trad_epochs += loss_Trad.item()
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            loss_U.backward(retain_graph=True)
            # loss_Trad.backward()
            # critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # 更新旧策略
        self.actor.load_state_dict(self.actor.state_dict())
        
        self.memory = {'local_state': [], 'global_state':[], 'actions': [], 'rewards': [], 'next_local_state': [],'next_global_state':[], 'dones': [], 'mask':[], 'next_mask':[]}
        
        return (actor_loss_epochs/self.epochs, loss_U/self.epochs, loss_trad_epochs/self.epochs)