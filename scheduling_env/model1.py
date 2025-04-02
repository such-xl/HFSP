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

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def size(self):
        return len(self.buffer)

# Actor 网络
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            
        )
    
    def forward(self, state):
        return F.softmax(self.fc(state), dim=-1)
    
    def get_action(self, state, available_actions=None):
        probs = self.forward(state)
        # 如果提供了可用动作掩码，则应用它
        if available_actions is not None:
            # 将不可用动作的概率设为0
            probs = probs * available_actions
            # 重新归一化概率
            probs = probs / (probs.sum() + 1e-10)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, dist.entropy()
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
    def __init__(self,state_dim,action_dim,lr,gamma,eplison,buff_size,divice = torch.device("cpu")):
        self.actor = Actor(state_dim,action_dim).to(device=divice)
        self.critic = Critic(state_dim).to(device=divice)
        self.device = divice
        self.gamma = gamma
        self.eplison = eplison
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.buffer = ReplayBuffer(buff_size)
        self.buff_size = buff_size

        #创建优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(),lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(),lr=lr)

    def take_action(self,state):
        states = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_prob = self.actor(state)
        action_prob = action_prob.detach().numpy()
        action = np.random.choice(self.action_dim,p=action_prob)
        return action,action_prob,action_prob[action]
    
    def train(self, lambda_weights=[0.5, 0.3, 0.2]):
        if len(self.buffer) < self.batch_size:
            return  

        # **从缓冲区采样**
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(device=self.device)
        actions = actions.to(device=self.device)    

        # 计算 Critic 目标值
        next_value_makespan, next_value_load_balance, next_value_energy = self.critic(next_states)
        value_makespan, value_load_balance, value_energy = self.critic(states)

        target_makespan = rewards + self.gamma * next_value_makespan.squeeze() * (1 - dones)
        target_load_balance = rewards + self.gamma * next_value_load_balance.squeeze() * (1 - dones)
        target_energy = rewards + self.gamma * next_value_energy.squeeze() * (1 - dones)

        # **计算加权 Advantage**
        total_advantage = (
            lambda_weights[0] * (target_makespan - value_makespan.squeeze()).detach() +
            lambda_weights[1] * (target_load_balance - value_load_balance.squeeze()).detach() +
            lambda_weights[2] * (target_energy - value_energy.squeeze()).detach()
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
       