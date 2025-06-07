import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(40)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.critic_U = nn.Linear(32, 1)
        self.critic_Wait = nn.Linear(32, 1)

    def forward(self, x, mask=None):
        # if mask is not None:
        #     x = x * mask.transpose(1, 2)
        shared_feature = self.shared_mlp(x)
        value_U = self.critic_U(shared_feature)
        value_wait = self.critic_Wait(shared_feature)
        return value_U, value_wait


class PPO:
    def __init__(
        self,
        local_state_dim,
        local_state_len,
        global_state_dim,
        global_state_len,
        act_dim,
        a_lr,
        c_lr,
        gamma,
        lmbda,
        eps,
        epochs,
        weights,  # 权重参数
        batch_size,
        path=None,
        device=torch.device("cpu"),
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.global_state_dim = global_state_dim
        self.global_state_len = global_state_len
        self.local_state_dim = local_state_dim
        self.local_state_len = local_state_len
        self.weights = torch.tensor(weights, dtype=torch.float).to(
            device
        )  # 添加权重参数
        self.actor = Actor(local_state_dim * local_state_len, act_dim).to(device=device)
        self.critic = Critic(global_state_dim * global_state_len).to(device=device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=c_lr)

        self.old_actor = Actor(
            local_state_dim * local_state_len,
            act_dim,
        ).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.obj_record = []
        self.memory = {
            "local_state": [],
            "global_state": [],
            "actions": [],
            "rewards": [],
            "next_local_state": [],
            "next_global_state": [],
            "dones": [],
            "mask": [],
            "next_mask": [],
        }

    def take_action(self, state, epsilon=0.01):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # if random.random() < epsilon:
        #     action = random.randint(0, 3)
        # else:
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action

    def store_transition(
        self,
        local_state,
        global_state,
        action,
        reward,
        next_local_state,
        next_global_state,
        done,
    ):
        self.memory["local_state"].append(local_state)
        self.memory["global_state"].append(global_state)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["next_local_state"].append(next_local_state)
        self.memory["next_global_state"].append(next_global_state)
        self.memory["dones"].append(done)

    def compute_advantage(self, td_delta):
        advantage = []
        adv = 0.0
        for delta in reversed(td_delta):
            adv = delta + self.gamma * self.lmbda * adv
            advantage.insert(0, adv)
        return torch.tensor(advantage, dtype=torch.float).to(self.device)

    def update(self):
        local_state = torch.tensor(self.memory["local_state"], dtype=torch.float).to(
            self.device
        )
        global_state = torch.tensor(self.memory["global_state"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(self.memory["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            self.memory["rewards"], dtype=torch.float32, device=self.device
        )  # 2个目标
        next_local_state = torch.tensor(
            self.memory["next_local_state"], dtype=torch.float
        ).to(self.device)
        next_global_state = torch.tensor(
            self.memory["next_global_state"], dtype=torch.float
        ).to(self.device)
        dones = torch.tensor(
            self.memory["dones"], dtype=torch.float32, device=self.device
        )

        old_log_probs = torch.log(self.actor(local_state).gather(1, actions)).detach()

        with torch.no_grad():
            value_U, value_Wait = self.critic(global_state)
            value_next_U, value_next_Wait = self.critic(next_global_state)
            value_U = value_U.squeeze(-1)
            value_Wait = value_Wait.squeeze(-1)
            value_next_U = value_next_U.squeeze(-1)
            value_next_Wait = value_next_Wait.squeeze(-1)

        # 计算 两个目标 TD 目标值
        rewards_1 = (rewards[:, 0] - rewards[:, 0].mean()) / (
            rewards[:, 0].std() + 1e-8
        )
        rewards_2 = (rewards[:, 1] - rewards[:, 1].mean()) / (
            rewards[:, 1].std() + 1e-8
        )
        td_target_U = rewards_1 + self.gamma * value_next_U * (1 - dones)
        td_target_Wait = rewards_2 + self.gamma * value_next_Wait * (1 - dones)
        advantage_U = self.compute_advantage(td_target_U - value_U)
        advantage_Wait = self.compute_advantage(td_target_Wait - value_Wait)
        # advantage = self.compute_advantage(td_target_U - value_U)
        # advantage = self.compute_advantage(td_target_Wait - value_Wait)

        # total_adv = abs(advantage_U.mean()) + abs(advantage_Wait.mean()) +1e-8
        # weight_U = abs(advantage_U.mean()) / total_adv
        # weight_Wait = abs(advantage_Wait.mean()) / total_adv
        # advantage = weight_U* advantage_U + weight_Wait * advantage_Wait
        advantage = self.weights[0] * advantage_U + self.weights[1] * advantage_Wait

        advantage = (advantage) / (advantage.std() + 1e-8)

        actor_loss_epochs = 0
        loss_U_epochs = 0
        loss_Wait_epochs = 0
        for _ in range(self.epochs):

            log_probs = torch.log(self.actor(local_state).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            value_U, value_Wait = self.critic(global_state)
            value_U = value_U.squeeze(-1)
            value_Wait = value_Wait.squeeze(-1)

            loss_fn = nn.MSELoss()
            loss_U = loss_fn(value_U, td_target_U.detach())
            loss_Wait = loss_fn(value_Wait, td_target_Wait.detach())

            actor_loss_epochs += actor_loss.item()
            loss_U_epochs += loss_U.item()
            loss_Wait_epochs += loss_Wait.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            # loss_U.backward(retain_graph=True)
            loss_Wait.backward(retain_graph=True)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 更新旧策略
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.memory = {
            "local_state": [],
            "global_state": [],
            "actions": [],
            "rewards": [],
            "next_local_state": [],
            "next_global_state": [],
            "dones": [],
            "mask": [],
            "next_mask": [],
        }

        self.obj_record.append(
            [td_target_U.mean().item(), td_target_Wait.mean().item()]
        )

        # return (actor_loss_epochs, 0, loss_Wait_epochs/self.epochs)
        return (actor_loss_epochs, loss_U / self.epochs, loss_Wait_epochs / self.epochs)

    def save_model(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path,weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.actor.eval()
        self.old_actor.eval()



# mini-batch

# data_len = local_state.shape[0]
# for _ in range(self.epochs):
#     indices = torch.randperm(data_len)
#     for i in range(0, data_len, self.batch_size):
#         batch_idx = indices[i:i + self.batch_size]

#         b_local_state = local_state[batch_idx]
#         b_global_state = global_state[batch_idx]
#         b_actions = actions[batch_idx]
#         b_old_log_probs = old_log_probs[batch_idx]
#         b_advantage = advantage[batch_idx]
#         b_td_target_U = td_target_U[batch_idx]
#         b_td_target_Wait = td_target_Wait[batch_idx]

#         log_probs = torch.log(self.actor(b_local_state).gather(1, b_actions))
#         ratio = torch.exp(log_probs - b_old_log_probs)
#         surr1 = ratio * b_advantage
#         surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * b_advantage
#         actor_loss = -torch.mean(torch.min(surr1, surr2))

#         value_U, value_Wait = self.critic(b_global_state)
#         value_U = value_U.squeeze(-1)
#         value_Wait = value_Wait.squeeze(-1)

#         loss_fn = nn.MSELoss()
#         loss_U = loss_fn(value_U, b_td_target_U.detach())
#         loss_Wait = loss_fn(value_Wait, b_td_target_Wait.detach())

#         actor_loss_epochs += actor_loss.item()
#         loss_U_epochs += loss_U.item()
#         loss_Wait_epochs += loss_Wait.item()

#         self.actor_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()
#         actor_loss.backward()
#         loss_U.backward(retain_graph=True)
#         loss_Wait.backward()
#         self.actor_optimizer.step()
#         self.critic_optimizer.step()
# # 更新旧策略
# num_updates = self.epochs * (data_len // self.batch_size + 1)
# self.old_actor.load_state_dict(self.actor.state_dict())
# self.memory = {'local_state': [], 'global_state':[], 'actions': [], 'rewards': [], 'next_local_state': [],'next_global_state':[], 'dones': [], 'mask':[], 'next_mask':[]}
# self.obj_record.append([td_target_U.mean().item(), td_target_Wait.mean().item()])
# return actor_loss_epochs/num_updates, loss_U / num_updates, loss_Wait/num_updates
