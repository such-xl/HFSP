import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(list(state)), action, reward, np.array(list(next_state)), done

    def size(self):
        return len(self.buffer)

# Actor 网络
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
    
    def forward(self, state):
        probs = F.softmax(self.linear(state), dim=-1)

        return Categorical(probs)
        
    
# Critic 网络（多任务）
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.critic_makespan = nn.Linear(128, 1)
        self.critic_load_balance = nn.Linear(128, 1)
        self.critic_energy = nn.Linear(128, 1)
    
    def forward(self, state):
        shared_repr = self.shared(state)
        return self.critic_makespan(shared_repr), self.critic_load_balance(shared_repr), self.critic_energy(shared_repr)

class PPOAgent:
    def __init__(self,obs_dim,obs_len,act_dim,global_state_dim,global_state_len,lr=3e-4, 
                 gamma=0.99,buffer_capacity=10000,device=torch.device("cpu")):
        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.act_dim = act_dim
        self.global_state_dim = global_state_dim
        self.global_state_len = global_state_len
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.actor = Actor(global_state_dim*global_state_len,act_dim).to(device=device)
        self.critic = Critic(global_state_dim).to(device=device)

        #创建优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(),lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(),lr=lr)


    def take_action(self,state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state_tensor)
            action = dist.sample()
        return action
    

    def update(self,batch):
        # **从缓冲区采样**

        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        # print(states.shape)
        # 计算 Critic 目标值
        value_makespan, value_load_balance, value_energy = self.critic(states)
        next_value_makespan, next_value_load_balance, next_value_energy = self.critic(next_states)
        
        value_makespan = next_value_makespan.max(dim=1)[0].squeeze(-1)
        value_load_balance = next_value_load_balance.mean(dim=1)[0].squeeze(-1)
        value_energy = next_value_energy.max(dim=1)[0].squeeze(-1)
        next_value_makespan = next_value_makespan.max(dim=1)[0].squeeze(-1)
        next_value_load_balance = next_value_load_balance.max(dim=1)[0].squeeze(-1)
        next_value_energy = next_value_energy.max(dim=1)[0].squeeze(-1)

        target_makespan = rewards + self.gamma * next_value_makespan.squeeze() * (1 - dones)
        target_load_balance = rewards + self.gamma * next_value_load_balance.squeeze() * (1 - dones)
        target_energy = rewards + self.gamma * next_value_energy.squeeze() * (1 - dones)

        # **计算加权 Advantage**
        total_advantage = (
            0.5 * (target_makespan - value_makespan.squeeze()).detach() +
            0.3 * (target_load_balance - value_load_balance.squeeze()).detach() +
            0.2 * (target_energy - value_energy.squeeze()).detach()
        )

        # **计算 Actor 损失**
        action_probs = self.actor(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        ratio = torch.exp(action_log_probs - action_log_probs.detach())

        loss_actor = -torch.min(
            ratio * total_advantage,  
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * total_advantage
        ).mean()

        # **计算 Critic 损失**
        loss_critic = (target_makespan - value_makespan.squeeze()).pow(2).mean() + \
                      (target_load_balance - value_load_balance.squeeze()).pow(2).mean() + \
                      (target_energy - value_energy.squeeze()).pow(2).mean()

        # **更新网络**
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        return loss_actor.item(), loss_critic.item()
       