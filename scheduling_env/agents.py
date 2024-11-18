import torch
import torch.nn.functional as F
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
            state_dim = model_params['state_dim'],
            x_dim = model_params['machine_dim'] + model_params['action_dim'],
            action_dim = model_params['action_dim'],
            num_heads = model_params['num_heads'],
            dropout =  model_params['dropout'],
        ).to(self.device)
        self.target_net = D3QN(
            state_dim = model_params['state_dim'],
            x_dim = model_params['machine_dim'] + model_params['action_dim'],
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


    def take_action(self,state,state_mask,machine_action,action_mask,step_done,):

        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)

        actions= []
        if np.random.random() < eps_threshold:
            i = 0
            action_mask_copy = np.copy(action_mask)
            while True:
                if i>=self.machine_seq_len or not machine_action[i].any():
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
                state = torch.as_tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
                state_mask = torch.as_tensor(state_mask, dtype=torch.bool).to(self.device).unsqueeze(0)

                mem = self.main_net.encoder(state,state_mask)
                machine_action = torch.as_tensor(machine_action, dtype=torch.float).to(self.device).unsqueeze(0)
                action_mask_copy = np.copy(action_mask)
                action_mask_copy = torch.as_tensor(action_mask_copy,dtype=torch.bool).to(self.device)
                machine_action_mask = torch.ones(1,self.machine_seq_len,dtype=torch.bool).to(self.device)
                i = 0
                while True:
                    if i>=self.machine_seq_len or not machine_action[0][i].any():
                        break
                    machine_action_mask[0,i] = False
                    q_value = self.main_net(mem,machine_action,machine_action_mask).squeeze(1)
                
                    # way 1:无效值屏蔽
                    q_value[~action_mask_copy[i].unsqueeze(0)] = -float('inf')
 
                    _,action = torch.max(q_value,dim=1)
                    machine_action[0][i][action+self.machine_dim] = 1
                    action = action.cpu().item()
                    if action != 31: # 非选择空闲动作
                        action_mask_copy[:,action] = False
                    actions.append(action)
                    i+=1
            machine_action = machine_action.cpu().numpy()
        return actions,machine_action

    def update(self, transition):
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

    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))
