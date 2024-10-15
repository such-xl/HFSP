import torch
import torch.nn.functional as F
import numpy as np
from .model import Actor
from .utils import StateNorm



class Agent():
    def __init__(self,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,
                 action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,learning_rate,gamma,target_update, device) -> None:
        self.actor = Actor(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads).to(device)
        self.target_actor = Actor(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate)
        self.job_input_dim  = job_input_dim
        self.job_seq_len = job_seq_len
        self.machine_seq_len = machine_seq_len
        self.action_dim = action_dim
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.gamma = gamma
        self.target_update = target_update
        self.device = device
        self.state_norm = StateNorm(job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        self.count = 0
        self.loss = 0
    def take_action(self,s_p_m,s_p_j,s_o_j,act_jobs,step_done):


        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * step_done / self.epsilon_decay)
        if np.random.random() < eps_threshold:
            action  = np.random.randint(0,len(act_jobs)+1) #最后一个动作代表空闲
        else:
            s_p_m = torch.tensor([s_p_m],dtype=torch.float).to(self.device)
            s_p_j,mask_spj = self.state_norm.job_seq_norm([s_p_j],0)
            s_o_j,mask_soj = self.state_norm.job_seq_norm([s_o_j],1)
            
            s_p_j = torch.tensor(s_p_j,dtype=torch.float).to(self.device)
            s_o_j = torch.tensor(s_o_j,dtype=torch.float).to(self.device)
            mask_spj = torch.tensor(mask_spj,dtype=torch.bool).to(self.device)
            mask_soj = torch.tensor(mask_soj,dtype=torch.bool).to(self.device)
            q_value = self.actor(s_p_m,s_p_j,s_o_j,mask_spj,mask_soj).view(-1)
            # way 1:无效值屏蔽
            '''
            action_value = torch.cat((q_value[:len(act_jobs)], q_value[-1].unsqueeze(0)))
            max_value, max_index = torch.max(action_value, dim=0)
            if max_index == len(act_jobs):
                action = self.action_dim-1
            else:
                action = max_index
            '''
            # way 2动作取余映射到合理范围
            _,action = torch.max(q_value,dim=0)
        return action
    def update(self,transition_dict):
        spms = torch.tensor(transition_dict['spms'],dtype=torch.float).to(self.device)
        spjs = torch.tensor(transition_dict['spjs'],dtype=torch.float).to(self.device)
        sojs = torch.tensor(transition_dict['sojs'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device) #[batsize,1]
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        nspms = torch.tensor(transition_dict['nspms'],dtype=torch.float).to(self.device)
        nspjs = torch.tensor(transition_dict['nspjs'],dtype=torch.float).to(self.device)
        nsojs = torch.tensor(transition_dict['nsojs'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        mask_spj = torch.tensor(transition_dict['mask_spj'],dtype=torch.bool).to(self.device)
        mask_soj = torch.tensor(transition_dict['mask_soj'],dtype=torch.bool).to(self.device)
        mask_nspj = torch.tensor(transition_dict['mask_nspj'],dtype=torch.bool).to(self.device)
        mask_nsoj = torch.tensor(transition_dict['mask_nsoj'],dtype=torch.bool).to(self.device)
        q_values = self.actor(spms,spjs,sojs,mask_spj,mask_soj).squeeze(1)# [batch_size,1,action_dim]->[batch_size,action_dim]

        if self.check_for_nan_inf(q_values):
            torch.save(spms,'spms.pt')
            torch.save(spjs,'spjs,pt')
            torch.save(sojs,'sojs.pt')
            torch.save(mask_spj,'mask_spj.pt')
            torch.save(mask_soj,'mask_soj.pt')
            raise ValueError(f"q_values contains NaN or Inf values")
        q_values = q_values.gather(1,actions) # [batch_size,1]
        
        '''
        print('actions:',actions.shape)
        print(actions)
        print('q_values_raw:',q_values.shape)
        print(q_values)
        print('q_values:',q_values.shape)
        print(q_values)
        '''
        with torch.no_grad():
            
            next_q_values = self.target_actor(nspms,nspjs,nsojs,mask_nspj,mask_nsoj).squeeze(1)# [64,1,30] ->[batch_size,action_dim]
            if self.check_for_nan_inf(next_q_values):
                torch.save(nspms,'nspms.pt')
                torch.save(nspjs,'nspjs,pt')
                torch.save(nsojs,'nsojs.pt')
                torch.save(mask_nspj,'mask_nspj.pt')
                torch.save(mask_nsoj,'mask_nsoj.pt')
                raise ValueError(f"next_q_values contains NaN or Inf values")
            # print('nqv:',next_q_values)
            # print(next_q_values)
            '''
            min_value = torch.finfo(next_q_values.dtype).min
            min_value = torch.tensor(min_value).to(self.device)
            mask_next_q_values = torch.where(mask_nspj,min_value.expand_as(next_q_values),next_q_values).to(self.device)
            max_nqv,_ = torch.max(mask_next_q_values,dim=1)
            max_nqv = max_nqv.view(-1,1)
            '''
            max_nqv,_ = torch.max(next_q_values,dim = 1)
            max_nqv = max_nqv.view(-1,1)

            q_targets = rewards + self.gamma*max_nqv*(1-dones)
            '''
            print('max_nqv:',max_nqv.shape)
            print(max_nqv)
            print('rewards:',rewards.shape)
            print(rewards)
            print('dones:',dones.shape)
            print(dones)
            print('q_targets:',q_targets.shape)
            print(q_targets)
            '''
        dqn_loss = F.smooth_l1_loss(q_values,q_targets)
        # print(q_values)
        # print(q_targets)
        # print('mse:',F.mse_loss(q_values,q_targets))
        # dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))
        self.loss += dqn_loss
        #print(dqn_loss.item())
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 200)
        self.optimizer.step()
        '''
        if (self.count+1) % self.target_update == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        '''
        actor_state_dict=self.actor.state_dict()
        target_state_dict = self.target_actor.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key]*self.tau + target_state_dict[key]*(1-self.tau)

        self.target_actor.load_state_dict(target_state_dict)
        if (self.count+1) % 1000 == 0:
            print('loss:',self.loss.item()/1000)
            self.loss = 0
        self.count += 1
    def save_model(self,path):
        torch.save(self.actor.state_dict(),path)
    def load_model(self,path):
        self.actor.load_state_dict(torch.load(path))

    def check_for_nan_inf(self,tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
            raise ValueError(f"{name} contains NaN or Inf values")
