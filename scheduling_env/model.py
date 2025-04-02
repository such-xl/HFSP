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
    
    def forward(self, x, mask):
        x = self.linear(x)
        x = x.masked_fill(mask,float('-inf'))
        x = F.softmax(x, dim=-1)
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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 任务特定 MLP
        self.critic_U = nn.Linear(64, 1)  # 能耗预测
        self.critic_trad = nn.Linear(64, 1)  # 时间预测

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
        self.actor = Actor(global_state_dim*global_state_len, act_dim).to(device=device)
        self.critic = Critic(global_state_dim).to(device=device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'mask':[], 'next_mask':[]}
    
    def take_action(self, state, mask):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mask = torch.tensor([mask], dtype=torch.bool).to(self.device)
        probs = self.actor(state,mask)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done, mask, next_mask):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
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
        if len(self.memory['states']) > self.batch_size:
            return 0,0
        print(len(self.memory['states']))
        states = torch.tensor(self.memory['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(self.memory['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32, device=self.device)  # 2个目标
        next_states = torch.tensor(self.memory['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.float32, device=self.device)
        mask = torch.tensor(self.memory['mask'], dtype=torch.bool).to(self.device)
        next_mask = torch.tensor(self.memory['next_mask'], dtype=torch.bool).to(self.device)
        rewards_U = rewards[:, 0]
        rewards_Trad = rewards[:, 1]
        old_log_probs = torch.log(self.actor(states,mask).gather(1, actions)).detach()
        
        with torch.no_grad():
            value_U, value_Trad = self.critic(states)
            value_next_U, value_next_Trad = self.critic(next_states)
            value_U = value_U.squeeze(-1)
            value_Trad = value_Trad.squeeze(-1)
            value_next_U = value_next_U.squeeze(-1)
            value_next_Trad = value_next_Trad.squeeze(-1)
        
         # 计算 两个目标 TD 目标值
        td_target_U = rewards_U + self.gamma*value_next_U*(1-dones)
        td_target_Trad = rewards_Trad + self.gamma*value_next_Trad*(1-dones)
        advantage_U = self.compute_advantage(td_target_U - value_U) 
        advantage_Trad = self.compute_advantage(td_target_Trad - value_Trad) 
        #动态调整权重
        # total_adv = abs(advantage_U.mean()) + abs(advantage_Trad.mean()) +1e-8
        # weight_U = abs(advantage_U.mean()) / total_adv
        # weight_Trad = abs(advantage_Trad.mean()) / total_adv
        advantage =  self.weights[0] * advantage_U + self.weights[1] * advantage_Trad 
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)

        actor_loss_epochs = 0
        critic_loss_epochs = 0
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states,mask).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            value_U, value_Trad = self.critic(states)
            value_U = value_U.squeeze(-1)
            value_Trad = value_Trad.squeeze(-1)
            critic_loss = torch.mean(
                    self.weights[0] * F.mse_loss(value_U, td_target_U.detach()) +
                    self.weights[1] * F.mse_loss(value_Trad, td_target_Trad.detach())
                )
            

            actor_loss_epochs += actor_loss.item()
            critic_loss_epochs +=critic_loss.item()
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        self.memory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [], 'mask':[], 'next_mask':[]}
        
        return (actor_loss_epochs/self.epochs, critic_loss_epochs/self.epochs)