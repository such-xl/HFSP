import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .model import D3QN
from .utils import StateNorm
# model_params = {
#     "state_dim": 18,
#     "machine_dim": 4,
#     "action_dim": 32,
#     "num_heads": 1,
#     "job_seq_len": 30,
#     "machine_seq_len": 16,
#     "dropout": 0.1,
# }
# train_params = {
#     "num_episodes": 100,
#     "batch_size": 512,
#     "learning_rate": 1e-6,
#     "epsilon_start": 1,
#     "epsilon_end": 1,
#     "epsilon_decay": 500,
#     "gamma": 1,
#     "tau": 0.005,
#     "target_update": 1000,
#     "buffer_size": 10_000,
#     "minimal_size": 1000,
#     "scale_factor": 0.01,
#     "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
#     "reward_type": 0,
# }
class Agent():
    def __init__(self,model_params,train_params) -> None:
        
        self.device = train_params['device']
        self.main_net = D3QN(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)

        self.target_net = D3QN(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)

        self.target_net.load_state_dict(self.main_net.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.main_net.parameters(), lr=train_params['learning_rate'])
        self.machine_seq_len = model_params['machine_seq_len']
        self.machine_dim = model_params['machine_dim']
        self.action_dim = model_params['action_dim']
        self.epsilon_start = train_params['epsilon_start']
        self.epsilon_end = train_params['epsilon_end']
        self.epsilon_decay = train_params['epsilon_decay']
        self.tau = train_params['tau']
        self.gamma = train_params['gamma']
        self.target_update = train_params['target_update']
        self.loss = 0
        self.count = 0


    def take_action(self,state,action_mask,step_done,):

        self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)
        
        if np.random.rand() < self.eps_threshold:
            available_actions = np.where(action_mask)[0]
            action = np.random.choice(available_actions)
            return action
        else:
            with torch.no_grad():
                state = torch.as_tensor(state).to(self.device,dtype=torch.float).unsqueeze(0)
                q_value = self.main_net(state).squeeze(0)
                action_mask = torch.as_tensor(action_mask,dtype=torch.bool).to(self.device)
                q_value[~action_mask] = -float('inf')
                _,action = torch.max(q_value,dim=-1)
                action = action.cpu().item()
                return action

    def update(self, transition):
        "state,actions,next_state,reward,done,next_action_mask"
        states = transition.states
        actions = transition.actions
        rewards = transition.rewards.squeeze(-1)
        dones = transition.dones.squeeze(-1)
        next_states = transition.next_states
        next_action_masks = transition.next_action_masks.to(torch.bool)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # print(next_states.shape)
        # print(next_action_masks.shape)

        q_values = self.main_net(states)
        actions = actions[:,0]
        Q = torch.sum(q_values * actions,dim=1)
        with torch.no_grad():
            next_q = self.main_net(next_states)
            next_q[~next_action_masks] = -float('inf')
            _, max_actions = next_q.max(dim=-1)
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.gather(1,max_actions.unsqueeze(-1)).squeeze(-1)
            Q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(Q,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.main_net.parameters(),100)
        self.optimizer.step()
        with torch.no_grad():
            self.loss += loss.item()
        actor_state_dict = self.main_net.state_dict()
        target_state_dict = self.target_net.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_state_dict)
        if (self.count + 1) % 500 == 0:
            print('loss:',self.loss/500)
            # print('epsilon:',self.eps_threshold)
            self.loss = 0
        self.count += 1

    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))
