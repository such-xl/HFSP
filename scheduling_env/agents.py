import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .model import D3QN,PolicyNet,QNet
from scipy.stats import beta
class D3QNAgent():
    def __init__(self,train_params,model_params) -> None:
        
        self.device = train_params['device']
        self.main_net = D3QN(
            state_dim=model_params['state_dim'],
            action_dim=model_params['action_dim']
        ).to(self.device)
        self.target_net = D3QN(
            state_dim=model_params['state_dim'],
            action_dim=model_params['action_dim']
        ).to(self.device)


        self.target_net.load_state_dict(self.main_net.state_dict())
        
        self.optimizer = torch.optim.AdamW(self.main_net.parameters(), lr=train_params['learning_rate'])

        self.action_dim = model_params['action_dim']
        self.epsilon_start = train_params['epsilon_start']
        self.epsilon_end = train_params['epsilon_end']
        self.epsilon_decay = train_params['epsilon_decay']
        self.tau = train_params['tau']
        self.gamma = train_params['gamma']
        self.target_update = train_params['target_update']
        self.loss = 0
        self.count = 0


    def take_action(self,state,step_done,):

        self.eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * step_done / self.epsilon_decay)
        
        if np.random.rand() < self.eps_threshold:
            return np.random.randint(0,self.action_dim)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state).to(self.device,dtype=torch.float).unsqueeze(0)
                q_value = self.main_net(state).squeeze(0)
                _,action = torch.max(q_value,dim=-1)
                action = action.cpu().item()
                return action

    def update(self, transition):
        "state,actions,next_state,reward,done"
        states = transition.states
        actions = transition.actions.to(torch.int64)
        rewards = transition.rewards.squeeze(-1)
        dones = transition.dones.squeeze(-1)
        next_states = transition.next_states

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(dones.shape)
        # print(next_states.shape)

        q_values = self.main_net(states)
        Q_values = q_values.gather(1,actions).squeeze(-1)
        with torch.no_grad():
            next_q = self.main_net(next_states)
            _, max_actions = next_q.max(dim=-1)
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.gather(1,max_actions.unsqueeze(-1)).squeeze(-1)
            Q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(Q_values,Q_targets)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.main_net.parameters(),100)
        self.optimizer.step()
        with torch.no_grad():
            self.loss += loss.item()
            self.count += 1
        actor_state_dict = self.main_net.state_dict()
        target_state_dict = self.target_net.state_dict()
        for key in actor_state_dict:
            target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_state_dict)
        if (self.count + 1) % 500 == 0:
            print('loss:',self.loss/500)
            self.loss = 0
        self.count += 1
        self.count+=1

        # if (self.count + 1) % self.target_update == 0:
        #     print('loss:',self.loss/self.target_update)
        #     actor_state_dict = self.main_net.state_dict()
        #     target_state_dict = self.target_net.state_dict()
        #     for key in actor_state_dict:
        #         target_state_dict[key] = actor_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        #     self.target_net.load_state_dict(target_state_dict)
        #     # print('epsilon:',self.eps_threshold)
        #     self.loss = 0

    def save_model(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load_model(self, path):
        self.main_net.load_state_dict(torch.load(path))

class SACAgent():
    def __init__(self,model_params,train_params) -> None:
        self.device = train_params['device']
        self.policy = PolicyNet(
            state_dim=model_params['state_dim'],
        ).to(self.device)

        self.q1 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.q2 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.target_q1 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.target_q2 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = torch.optim.AdamW(self.q1.parameters(), lr=train_params['learning_rate'])
        self.q2_optimizer = torch.optim.AdamW(self.q2.parameters(), lr=train_params['learning_rate'])

        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=train_params['learning_rate'])
        self.target_entropy = -1  # 目标熵，通常设置为负的动作维度
        self.log_alpha = torch.tensor(0.0, requires_grad=True)  # α 的对数形式，用于优化
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=6e-5)  # 优化器

        self.alpha = self.log_alpha.exp().item()  # Initialize alpha
        self.tau = train_params['tau']
        self.gamma = train_params['gamma']
        self.target_update = train_params['target_update']
        self.loss = 0
        self.count = 0

    def take_action(self,state,deterministic = False):
        with torch.no_grad():
            state = torch.as_tensor(state).to(self.device,dtype=torch.float)
            alpha, beta = self.forward(state)
            action = beta.rvs(alpha.detach().numpy(), beta.detach().numpy()).cpu().item()
            return action
    def get_batch_actions(self,state):
        probs = self.policy(state)
        actions = torch.distributions.Categorical(probs).sample()
        return actions
    def update(self,transition):
        
        states = transition.states
        actions = transition.actions.to(torch.int64)
        rewards = transition.rewards.squeeze(-1)
        dones = transition.dones.squeeze(-1)
        next_states = transition.next_states

        probs = self.policy(states)
        policy_loss = (probs * (self.alpha * torch.log(probs + 1e-9) - q_values)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs,dim=-1).mean()
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()
        # Soft update target networks

        self.count += 1
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

class Agents():
    def __init__(self,train_params,model_params) -> None:
        self.agents = [SACAgent(model_params,train_params) for _ in range(model_params['machine_seq_len'])]
        self.tau = train_params['tau']
        self.device = train_params['device']
        self.gamma = train_params['gamma']
        self.q1 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.q2 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.target_q1 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)
        self.target_q2 = QNet(
            state_dim=model_params['state_dim'],
            machine_dim=model_params['machine_dim'],
            action_dim=model_params['action_dim'],
        ).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = torch.optim.AdamW(self.q1.parameters(), lr=train_params['learning_rate'])
        self.q2_optimizer = torch.optim.AdamW(self.q2.parameters(), lr=train_params['learning_rate'])

    def take_actions(self,state) -> list:
        actions = []
        state = torch.as_tensor(state).to(dtype=torch.float,device = self.device)
        for i,agent in enumerate(self.agents):
            machine_state = [[[1 if i==j else 0 for j in range(16)]]]
            machine_state = torch.as_tensor(machine_state).to(dtype=torch.float,device = self.device)
            machine_state = machine_state.expand(state.size(0),1,-1)
            s = torch.cat([state,machine_state],dim=1)
            actions.append(agent.take_action(s))
        return actions
    def get_batch_actions(self,states):
        actions = torch.zeros([states.size(0),len(self.agents)],dtype=torch.int64,device = self.device)
        for i,agent in enumerate(self.agents):
            machine_state = [[[1 if i==j else 0 for j in range(16)]]]
            machine_state = torch.as_tensor(machine_state).to(dtype=torch.float,device = self.device)
            machine_state = machine_state.expand(states.size(0),1,-1)
            s = torch.cat([states,machine_state],dim=1)
            actions[:,i:i+1] = agent.get_batch_actions(s).unsqueeze(-1)
        return actions
    def update(self,transition):
        states = transition.states
        actions = transition.actions
        rewards = transition.rewards
        dones = transition.dones
        next_states = transition.next_states
        
        with torch.no_grad():
            next_actions = self.get_batch_actions(next_states).to(dtype=torch.int64,device = self.agents[0].device)
            next_q1 = self.target_q1(next_states,next_actions)
            next_q2 = self.target_q2(next_states,next_actions)
            min_next_q = torch.min(next_q1,next_q2)
            Q_targets = rewards + 1 * (1 - dones) * min_next_q

        q1 = self.q1(states,actions)
        q2 = self.q2(states,actions)

        q1_loss = F.smooth_l1_loss(q1,Q_targets)
        q2_loss = F.smooth_l1_loss(q2,Q_targets)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        with torch.no_grad():
            q_values = torch.min(self.q1(states,actions),self.q2(states,actions))        
        for i,agent in enumerate(self.agents):
            machine_state = [[[1 if i==j else 0 for j in range(16)]]]
            machine_state = torch.as_tensor(machine_state).to(dtype=torch.float,device = self.device)
            machine_state = machine_state.expand(states.size(0),1,-1)
            s = torch.cat([states,machine_state],dim=1)
            agent.update(s.clone(),q_values.clone())
        
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)