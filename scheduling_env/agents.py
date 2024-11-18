from typing import NamedTuple
import time
import torch
import torch.nn.functional as F
import numpy as np
from .model import Actor, Actora
from .utils import StateNorm

class Agent():
    def __init__(self, job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim,
                 action_dim, num_heads, job_seq_len, machine_seq_len, epsilon_start, epsilon_end, epsilon_decay, tau,
                 learning_rate, gamma, target_update, device) -> None:
        self.actor = Actor(job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim, action_dim,
                           num_heads).to(device)
        self.target_actor = Actor(job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim, action_dim,
                                  num_heads).to(device)
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate)
        print(learning_rate)
        self.job_input_dim = job_input_dim
        self.job_seq_len = job_seq_len
        self.machine_seq_len = machine_seq_len
        self.machine_dim = machine_input_dim
        self.action_dim = action_dim
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.gamma = gamma
        self.target_update = target_update
        self.device = device
        self.count = 0
        self.loss = 0


    def take_action(self,state,state_mask,machine_action,action_mask,step_done,):

        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)

        actions= []
        if np.random.random() < eps_threshold:
            i = 0
            action_mask_copy = np.copy(action_mask)
            while True:
                if i>=15 or not machine_action[i].any():
                    break
                available_actions = np.where(action_mask_copy[i])[0]
                action = np.random.choice(available_actions)
                actions.append(action)
                machine_action[i][action+self.machine_dim] = 1
                if action != self.action_dim-1: # 非空闲动作
                   action_mask_copy[:,action] = False  # 其他智能体不再可选该动作
                i += 1
        else:
            with torch.no_grad():
                machine_state = torch.as_tensor(machine_state, dtype=torch.float).to(self.device).unsqueeze(0)
                job_state = torch.as_tensor(job_state, dtype=torch.float).to(self.device).unsqueeze(0)
                machine_mask = torch.as_tensor(machine_mask, dtype=torch.bool).to(self.device).unsqueeze(0)
                job_mask = torch.as_tensor(job_mask, dtype=torch.bool).to(self.device).unsqueeze(0)

                q_value = self.actor(machine_state,job_state,machine_mask,job_mask).squeeze(1)

                # way 1:无效值屏蔽
                action_mask = torch.as_tensor(action_mask,dtype=torch.bool).to(self.device).unsqueeze(0)
                q_value[~action_mask] = -float('inf')
            
                _,action = torch.max(q_value,dim=1)
                action = action.cpu().item()
        return actions,machine_action

    def update(self, transition):
        gst = time.time()
        machine_states = transition.machine_states
        job_states = transition.job_states
        machine_masks = transition.machine_masks.to(torch.bool)
        job_masks = transition.job_masks.to(torch.bool)
        action_masks = transition.action_masks.to(torch.bool)
        actions = transition.actions.to(torch.int64)
        next_machine_states = transition.next_machine_states
        next_job_states = transition.next_job_states
        next_machine_masks = transition.next_machine_masks.to(torch.bool)
        next_job_masks = transition.next_job_masks.to(torch.bool)
        next_action_masks = transition.next_action_masks.to(torch.bool)
        rewards = transition.rewards
        dones = transition.dones
        '''
        print('machine_states:',machine_states.shape)
        print('job_states:',job_states.shape)
        print('machine_masks:',machine_masks.shape)
        print('job_masks:',job_masks.shape)
        print('action_masks:',action_masks.shape)
        print('actions:',actions.shape)
        print('next_machine_states:',next_machine_states.shape)
        print('next_job_states:',next_job_states.shape)
        print('next_machine_masks:',next_machine_masks.shape)
        print('next_job_masks:',next_job_masks.shape)
        print('next_action_masks:',next_action_masks.shape)
        print('rewards:',rewards.shape)
        print('dones:',dones.shape)
        '''

        q_values = self.actor(machine_states,job_states,machine_masks,job_masks).squeeze(1)  # [batch_size,1,action_dim]->[batch_size,action_dim]
        q_values = q_values.gather(1, actions)  # [batch_size,1]
        with torch.no_grad():
            next_q_values = self.target_actor(next_machine_states,next_job_states,next_machine_masks,next_job_masks).squeeze(1)  # [batch_size,1,action_dim] ->[batch_size,action_dim]
            next_q_values[~next_action_masks] = -float('inf')
            max_nqv,_ = torch.max(next_q_values,dim=1)

            max_nqv = max_nqv.view(-1,1)
            q_targets = rewards + self.gamma * max_nqv * (1 - dones)
        dqn_loss = F.smooth_l1_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        with torch.no_grad():
            self.loss += dqn_loss
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 200)
        self.optimizer.step()
        actor_state_dict = self.actor.state_dict()
        target_state_dict = self.target_actor.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_actor.load_state_dict(target_state_dict)
        if (self.count + 1) % 1000 == 0:
            print('loss:',self.loss.item()/1000)
            self.loss = 0
        self.count += 1
        tt = time.time()-gst
        return tt

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))


    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
