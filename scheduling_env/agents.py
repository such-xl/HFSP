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
                    machine_action_mask[:,i] = False
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
        states = transition.states
        state_masks = transition.state_masks.to(torch.bool)
        machine_actions = transition.machine_actions
        rewards = transition.rewards
        dones = transition.dones
        next_states = transition.next_states
        next_state_masks = transition.next_state_masks.to(torch.bool)

        print('states:',states.shape)
        print('state_masks:',state_masks.shape)
        print('machine_actions:',machine_actions.shape)
        print('rewards:',rewards.shape)
        print('dones:',dones.shape)
        print('next_states:',next_states.shape)
        print('next_state_masks:',next_state_masks.shape)

        pass
        mem = self.main_net.encoder(states,state_masks)
<<<<<<< HEAD
        machine_actions_copy = machine_actions.clone()
        machine_actions_copy[:,:,self.machine_dim:] = 0
        key_padding_mask = torch.ones(machine_actions.size(0),self.machine_seq_len,dtype=torch.bool).to(self.device)
        for i in range(self.machine_seq_len):
            key_padding_mask[:,i] = False
            q_values = self.main_net(mem,machine_actions_copy,key_padding_mask)
            q_values = q_values[machine_actions[:,i,self.machine_dim:].to(torch.bool)].unsqueeze(1)
            pass
            
=======
        for i in range(self.machine_seq_len):
            q_vlaues = self.main_net(mem,machine_actions[:,i,:])
        
        q_values = self.actor(machine_states,job_states,machine_masks,job_masks).squeeze(1)  # [batch_size,1,action_dim]->[batch_size,action_dim]
        q_values = q_values.gather(1, actions)  # [batch_size,1]
>>>>>>> d448acc22b31b22ebcd6783f2400f796bd298978
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
