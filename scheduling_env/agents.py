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
            state_seq_len = model_params['job_seq_len'],
            embedding_dim = model_params['state_embedding_dim'],
            machine_state_dim = model_params['machine_state_dim'],
            action_dim = model_params['action_dim'],
            num_heads = model_params['num_heads'],
            dropout =  model_params['dropout'],
        ).to(self.device)

        self.target_net = D3QN(
            state_seq_len = model_params['job_seq_len'],
            embedding_dim = model_params['state_embedding_dim'],
            machine_state_dim = model_params['machine_state_dim'],
            action_dim = model_params['action_dim'],
            num_heads = model_params['num_heads'],
            dropout =  model_params['dropout'],
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
        
        if True or np.random.rand() < self.eps_threshold:
            available_actions = np.where(action_mask)[0]
            action = np.random.choice(available_actions)
            return action
        else:
            with torch.no_grad():
                state = torch.as_tensor(state).to(self.device).unsqueeze(0)
                machine_state = torch.as_tensor(machine_state).to(self.device).unsqueeze(0)
                mem = self.main_net.encoder(state)
                q_value = self.main_net(mem,machine_state).squeeze(0)
                action_mask_copy = np.copy(action_mask)
                action_mask_copy = torch.as_tensor(action_mask_copy,dtype=torch.bool).to(self.device)
                for i in range(machine_state.size(1)):
                    q_value_i = q_value[i]
                    q_value_i[~action_mask_copy[i]] = -float('inf')
                    _,action = torch.max(q_value_i,dim=0)
                    action = action.cpu().item()
                    action.append(action)
                    if action != 20:
                        action_mask_copy[:,action] = False
        return action

    def update(self, transition):
        states = transition.states
        machine_state = transition.machine_state
        actions = transition.actions.to(torch.int64)
        action_masks = transition.action_masks.to(torch.bool)
        rewards = transition.rewards
        dones = transition.dones
        next_states = transition.next_states
        next_machine_states = transition.next_machine_states
        next_action_masks = transition.next_action_masks.to(torch.bool)

        mem = self.main_net.encoder(states)
        q_values = self.main_net(mem,machine_state)
        Q_all = torch.zeros(machine_state.size(0),machine_state.size(1), device=q_values.device)

        for i in range(machine_state.size(1)):
            q_i = q_values[:, i, :]
            action = actions[:, i:i+1]
            qa_i = q_i.gather(1, action)
            Q_all[:, i:i+1] = qa_i  # 将结果填入预分配的张量
        # way 1:对Q_all进行Loss
        # way 2:对Q_all的均值进行Loss

        with torch.no_grad():
            next_mem = self.target_net.encoder(next_states)
            next_q_values = self.main_net(next_mem,next_machine_states)
            next_Q_all = None
            for i in range(next_machine_states.size(1)):
                n_qi = next_q_values[:,i,:]
                n_max_qi,_ = n_qi.max(dim=1)
                n_max_qi.unsqueeze_(-1)
                if next_Q_all is None:
                    next_Q_all = n_max_qi
                else:
                    next_Q_all = torch.cat([next_Q_all,n_max_qi],dim = 1)

            # for i in range(self.machine_seq_len):
            #     if not next_machine_actions[:,i].any():
            #         break
            #     next_key_padding_mask[:,i] = False
            #     next_q_values = self.target_net(next_mem,next_machine_actions_copy,next_key_padding_mask)
            #     next_machine_actions_copy[:,i] = next_machine_actions[:,i]
            #     actions = next_machine_actions[:,i,1:]
            #     max_values = next_q_values.gather(1,actions)
            #     mask_any = next_machine_actions[:,i,:].any(dim=-1, keepdim=True)
            #     next_q = torch.zeros(next_q_values.size(0), 1).to(self.device)
            #     next_q[mask_any] = max_values[mask_any]
            #     next_Q_all += next_q
            # Q_targets = rewards + self.gamma * next_Q_all * (1 - dones)
        # 计算q_tot_target
        with torch.no_grad():
            q_tot_target = self.main_net.mixing_network(next_Q_all,next_mem.clone())
            q_tot_target = rewards + self.gamma * (1 - dones) * q_tot_target
        # 计算q_tot
        # mem_copy = mem.clone()
        q_tot = self.main_net.mixing_network(Q_all,mem)
                # Loss and backpropagation
        loss = nn.SmoothL1Loss()(q_tot, q_tot_target)
        # print("loss:",loss.item())
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.main_net.parameters(),100)
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
            print('epsilon:',self.eps_threshold)
            self.loss = 0
        self.count += 1

    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))
