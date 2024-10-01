import torch
import torch.nn.functional as F
import numpy as np
from .model import Actor
from .utils import StateNorm
class Agent():
    def __init__(self,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,
                 action_dim,num_heads,job_seq_len,machine_seq_len,epsilon,learning_rate,gamma,target_update, device) -> None:
        self.actor = Actor(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads).to(device)
        self.target_actor = Actor(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.job_input_dim  = job_input_dim
        self.job_seq_len = job_seq_len
        self.machine_seq_len = machine_seq_len
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_update = target_update
        self.device = device
        self.state_norm = StateNorm(job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        self.count = 0
    def take_action(self,s_p_m,s_p_j,s_o_j,act_jobs):
        if np.random.random() < self.epsilon:
            action  = np.random.randint(-1,len(act_jobs))
        else:
            s_p_m = torch.tensor([s_p_m],dtype=torch.float).to(self.device)
            s_p_j,mask_spj = self.state_norm.job_seq_norm([s_p_j],0)
            s_o_j,mask_soj = self.state_norm.job_seq_norm([s_o_j],1)
            
            s_p_j = torch.tensor(s_p_j,dtype=torch.float).to(self.device)
            s_o_j = torch.tensor(s_o_j,dtype=torch.float).to(self.device)
            mask_spj = torch.tensor(mask_spj,dtype=torch.bool).to(self.device)
            mask_soj = torch.tensor(mask_soj,dtype=torch.bool).to(self.device)
            q_value = self.actor(s_p_m,s_p_j,s_o_j,mask_spj,mask_soj).view(-1)
            action_value = torch.cat((q_value[:len(act_jobs)], q_value[-1].unsqueeze(0)))
            max_value, max_index = torch.max(action_value, dim=0)
            if max_index == len(act_jobs):
                action = self.action_dim-1
            else:
                action = max_index
        return action
    def update(self,transition_dict):
        spms = torch.tensor(transition_dict['spms'],dtype=torch.float).to(self.device)
        spjs = torch.tensor(transition_dict['spjs'],dtype=torch.float).to(self.device)
        sojs = torch.tensor(transition_dict['sojs'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        nspms = torch.tensor(transition_dict['nspms'],dtype=torch.float).to(self.device)
        nspjs = torch.tensor(transition_dict['nspjs'],dtype=torch.float).to(self.device)
        nsojs = torch.tensor(transition_dict['nsojs'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        mask_spj = torch.tensor(transition_dict['mask_spj'],dtype=torch.bool).to(self.device)
        mask_soj = torch.tensor(transition_dict['mask_soj'],dtype=torch.bool).to(self.device)
        mask_nspj = torch.tensor(transition_dict['mask_nspj'],dtype=torch.bool).to(self.device)
        mask_nsoj = torch.tensor(transition_dict['mask_nsoj'],dtype=torch.bool).to(self.device)
        q_values = self.actor(spms,spjs,sojs,mask_spj,mask_soj).gather(2,actions).squeeze(-1) #[64,1]
        # print(actions.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # print(mask_spj.shape)
        # print(mask_soj.shape)
        # print(mask_nspj.shape)
        # print(mask_nsoj.shape)
        next_q_values = self.target_actor(nspms,nspjs,nsojs,mask_nspj,mask_nsoj).squeeze(1)# [64,1,30] ->[64,30]
        # if torch.isnan(next_q_values).any():
        #     print("NaN detected in next_q_values")
        min_value = torch.finfo(next_q_values.dtype).min
        min_value = torch.tensor(min_value).to(self.device)
        mask_next_q_values = torch.where(mask_nspj,min_value.expand_as(next_q_values),next_q_values).to(self.device)
        max_nqv,_ = torch.max(mask_next_q_values,dim=1)
        max_nqv = max_nqv.view(-1,1)
        q_targets = rewards + self.gamma*max_nqv*(1-dones)
        dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))
        #print(dqn_loss.item())
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        self.count += 1