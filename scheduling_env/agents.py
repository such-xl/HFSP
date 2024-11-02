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
        self.action_dim = action_dim
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.gamma = gamma
        self.target_update = target_update
        self.device = device
        self.state_norm = StateNorm(job_input_dim, job_seq_len, machine_input_dim, machine_seq_len,action_dim)
        self.count = 0
        self.loss = 0

    def take_action(self,machine_state,machine_mask,job_state,job_mask,act_jobs, action_mask,step_done,):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)
        if np.random.random() < eps_threshold:
            action = np.random.randint(0, len(act_jobs) + 1)  # 最后一个动作代表空闲
        else:
            with torch.no_grad():
                machine_state = torch.as_tensor(machine_state, dtype=torch.float).to(self.device).unsqueeze(0)
                job_state = torch.as_tensor(job_state, dtype=torch.float).to(self.device).unsqueeze(0)
                machine_mask = torch.as_tensor(machine_mask, dtype=torch.bool).to(self.device).unsqueeze(0)
                job_mask = torch.as_tensor(job_mask, dtype=torch.bool).to(self.device).unsqueeze(0)

                q_value = self.actor(machine_state,job_state,machine_mask,job_mask).squeeze(1)

                # way 1:无效值屏蔽
                action_mask = torch.as_tensor(action_mask,dtype=torch.float).to(self.device).unsqueeze(0)
                mask_q_value = q_value * action_mask
            
                _,action = torch.max(mask_q_value,dim=1)
                action = action.cpu().item()
        return action

    def update(self, transition):
        gst = time.time()
        machine_states = transition.machine_states
        job_states = transition.job_states
        machine_masks = transition.machine_masks.to(torch.bool)
        job_masks = transition.job_masks.to(torch.bool)
        action_masks = transition.action_masks
        actions = transition.actions.to(torch.int64)
        next_machine_states = transition.next_machine_states
        next_job_states = transition.next_job_states
        next_machine_masks = transition.next_machine_masks.to(torch.bool)
        next_job_masks = transition.next_job_masks.to(torch.bool)
        next_action_masks = transition.next_action_masks
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

        gt = time.time()-gst
        tst = time.time()
        q_values = self.actor(machine_states,job_states,machine_masks,job_masks).squeeze(1)  # [batch_size,1,action_dim]->[batch_size,action_dim]
        
        q_values = q_values.gather(1, actions)  # [batch_size,1]
        with torch.no_grad():
            next_q_values = self.target_actor(next_machine_states,next_job_states,next_machine_masks,next_job_masks).squeeze(1)  # [batch_size,1,action_dim] ->[batch_size,action_dim]
            mask_next_q_values = next_q_values * next_action_masks
            max_nqv,_ = torch.max(mask_next_q_values,dim=1)

            max_nqv = max_nqv.view(-1,1)
            q_targets = rewards + self.gamma * max_nqv * (1 - dones)
        dqn_loss = F.smooth_l1_loss(q_values, q_targets)
        self.loss += dqn_loss
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 200)
        self.optimizer.step()

        actor_state_dict = self.actor.state_dict()
        target_state_dict = self.target_actor.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        tt = time.time()-tst

        self.target_actor.load_state_dict(target_state_dict)
        if (self.count + 1) % 1000 == 0:
            print('loss:', self.loss.item() / 1000)
            self.loss = 0
        self.count += 1
        return gt,tt

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))


class Agenta():
    def __init__(self, job_input_dim, job_hidden_dim1, job_hidden_dim2, machine_input_dim, machine_hidden_dim,
                 cfc_hidden_dim1, cfc_hidden_dim2, action_dim, job_seq_len, machine_seq_len, epsilon_start,
                 epsilon_end, epsilon_decay, tau, learning_rate, gamma, target_update, device) -> None:

        self.actor = Actora(job_seq_len, job_input_dim, job_hidden_dim1, job_hidden_dim2, machine_seq_len,
                            machine_input_dim, machine_hidden_dim, cfc_hidden_dim1, cfc_hidden_dim2, action_dim).to(
            device)

        self.target_actor = Actora(job_seq_len, job_input_dim, job_hidden_dim1, job_hidden_dim2, machine_seq_len,
                                   machine_input_dim, machine_hidden_dim, cfc_hidden_dim1, cfc_hidden_dim2,
                                   action_dim).to(device)
        print(learning_rate)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.job_input_dim = job_input_dim
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
        self.state_norm = StateNorm(job_input_dim, job_seq_len, machine_input_dim, machine_seq_len)
        self.count = 0
        self.loss = 0

    def take_action(self, s_p_m, s_p_j, s_o_j, act_jobs, step_done):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)
        if np.random.random() < eps_threshold:
            return np.random.randint(0, len(act_jobs) + 1)  # 最后一个动作代表空闲
        else:
            s_p_m = torch.tensor([s_p_m], dtype=torch.float).to(self.device)
            s_p_j, mask_spj = self.state_norm.job_seq_norm(s_p_j, 0)
            s_o_j, mask_soj = self.state_norm.job_seq_norm(s_o_j, 1)

            s_p_j = torch.tensor(s_p_j, dtype=torch.float).to(self.device).unsqueeze(0)
            s_o_j = torch.tensor(s_o_j, dtype=torch.float).to(self.device).unsqueeze(0)
            mask_spj = torch.tensor(mask_spj, dtype=torch.bool).to(self.device).unsqueeze(0)
            mask_soj = torch.tensor(mask_soj, dtype=torch.bool).to(self.device).unsqueeze(0)
            q_value = self.actor(s_p_m, s_p_j, s_o_j, mask_spj, mask_soj).view(-1)
            # way 1:无效值屏
            action_value = torch.cat((q_value[:len(act_jobs)], q_value[-1].unsqueeze(0)))
            max_value, max_index = torch.max(action_value, dim=0)
            return max_index
            '''
            # way 2动作取余映射到合理范围
            _, action = torch.max(q_value, dim=0)
            '''
        return action.cpu().item()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    def update(self, transition):
        gst = time.time()
        spms =transition.spms
        spms = transition.spms
        spjs = transition.spjs
        sojs=transition.sojs
        actions=transition.actions.to(torch.int64)
        rewards=transition.rewards
        nspms=transition.nspms
        nspjs=transition.nspjs
        nsojs=transition.nsojs
        dones=transition.dones
        mask_spj=transition.mask_spj
        mask_soj=transition.mask_soj
        mask_nspj= transition.mask_nspj
        mask_nsoj= transition.mask_nsoj
        gt = time.time()-gst
        tst = time.time()
        q_values = self.actor(spms, spjs, sojs, mask_spj, mask_soj).squeeze(1)  # [batch_size,1,action_dim]->[batch_size,action_dim]
        q_values = q_values.gather(1, actions)  # [batch_size,1]
        with torch.no_grad():
            next_q_values = self.target_actor(nspms, nspjs, nsojs, mask_nspj, mask_nsoj).squeeze(1)  # [64,1,30] ->[batch_size,action_dim]
            """
            max_nqv, _ = torch.max(next_q_values, dim=1)
            max_nqv = max_nqv.view(-1, 1)
            """
            masked_next_q_values = next_q_values.where(mask_nspj, torch.full_like(next_q_values, float('-inf')))

            # 计算屏蔽后的最大值
            max_nqv, _ = torch.max(masked_next_q_values, dim=1)
            max_nqv = max_nqv.view(-1, 1)
            q_targets = rewards + self.gamma * max_nqv * (1 - dones)
        dqn_loss = F.smooth_l1_loss(q_values, q_targets)
        self.loss += dqn_loss
        # print(dqn_loss.item())
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 200)
        self.optimizer.step()

        actor_state_dict = self.actor.state_dict()
        target_state_dict = self.target_actor.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)

        self.target_actor.load_state_dict(target_state_dict)
        tt = time.time()-tst
        if (self.count + 1) % 1000 == 0:
            print('loss:', self.loss.item() / 1000)
            self.loss = 0
        self.count += 1
        return gt,tt
    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))

    def check_for_nan_inf(self, tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
            raise ValueError(f"{name} contains NaN or Inf values")
