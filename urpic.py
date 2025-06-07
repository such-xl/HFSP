import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from collections import deque
import random
from scheduling_env.training_env import TrainingEnv
from params import PARAMS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)


# set_seed(42)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state, action_dim: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # 从第一个维度开始展平 (保留 batch 维度)
        self.shared = nn.Sequential(
            nn.Linear(state, 64),  # 输入维度是 5 * 6 = 30
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )
        self.U_head = nn.Linear(32, action_dim)
        self.Slack_head = nn.Linear(32, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        x = self.shared(x)
        return self.U_head(x), self.Slack_head(x)

class DDQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(DDQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class SHQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SHQNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class SHDQN:
    def __init__(
        self,
        state_dim,
        state_len,
        action_dim,
        weights=[0.5, 0.5],
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.9999,
        buffer_size=10000,
        batch_size=64,
        target_update_interval=10,
    ):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_step_counter = 0
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

        self.q_net = SHQNetwork(state_dim * state_len, action_dim).to(device)
        self.target_net = SHQNetwork(state_dim * state_len, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        current_q = self.q_net(states).gather(1, actions).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = (
                self.weights[0] * rewards[:, 0]
                + self.weights[1] * rewards[:, 1]
                + self.gamma * next_q * (1 - dones)
            )

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path: str):
        torch.save({"q_net": self.q_net.state_dict()}, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_net.eval()

class DMDDQN:
    def __init__(
        self,
        local_state_dim,
        local_state_len,
        action_dim,
        gamma=0.99,
        batch_size=64,
        tau=0.005,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.005,
        weights=[0.5, 0.5],
        lr=1e-3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DDQNetwork(local_state_dim * local_state_len, action_dim).to(self.device)
        self.target_network = DDQNetwork(local_state_dim * local_state_len, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_weights = torch.tensor(weights).to(self.device)
        self.replay_buffer = ReplayBuffer(1000)
        self.action_dim = action_dim

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def store_transition(self, *args):
        self.replay_buffer.add(*args)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        s, a, r, next_s, d = self.replay_buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)  # 2个目标
        next_s = torch.FloatTensor(next_s).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
    

        # 合成标量奖励
        r_total = (r @ self.reward_weights).unsqueeze(1)
        q_eval = self.q_network(s).gather(1, a.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.q_network(next_s).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_s).gather(1, next_actions)
            q_target = r_total + (1 - d.unsqueeze(1)) * self.gamma * next_q

        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 target 网络
        for t_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            t_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * t_param.data)

        # 衰减 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path: str):
        torch.save({"q_net": self.q_network.state_dict()}, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_net"])
        self.q_network.eval()

class DDQN:
    def __init__(
        self,
        state_dim,
        state_len,
        action_dim,
        weights=[0.5, 0.5],
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.9999,
        buffer_size=10000,
        batch_size=64,
        target_update_interval=10,
    ):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_step_counter = 0
        self.weights = torch.tensor(weights).to(device)
        self.q_net = QNetwork(state_dim * state_len, action_dim).to(device)
        self.target_net = QNetwork(state_dim * state_len, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.target_net.load_state_dict(self.q_net.state_dict())

    def take_action(self, state):
        if random.random() < self.epsilon_min:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            U, Wait = self.q_net(state)
            q_values = self.weights[0] * U + self.weights[1] * Wait
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        self.learn_step_counter += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 2个目标
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        U, Wait = self.q_net(states)
        current_q_U = U.gather(1, actions).squeeze(-1)
        current_q_Wait = Wait.gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_U_online, next_Slack_online = self.q_net(next_states)
            next_actions = (
                self.weights[0] * next_U_online + self.weights[1] * next_Slack_online
            ).argmax(1)

            next_U_target, next_Slack_target = self.target_net(next_states)
            next_q_U = next_U_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            next_q_Slack = next_Slack_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()

            target_U = rewards[:, 0] + self.gamma * next_q_U * (1 - dones)
            target_Slack = rewards[:, 1] + self.gamma * next_q_Slack * (1 - dones)

        loss_U = F.mse_loss(current_q_U, target_U)
        loss_Slack = F.mse_loss(current_q_Wait, target_Slack)
        total_loss = self.weights[0] * loss_U + self.weights[1] * loss_Slack

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.learn_step_counter % self.target_update_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path: str):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_net.eval()

class THDQN:
    def __init__(
        self,
        state_dim,
        state_len,
        action_dim,
        weights=[0.5, 0.5],
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.005,
        epsilon_decay=0.9999,
        buffer_size=10000,
        batch_size=64,
        target_update_interval=10,
    ):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_step_counter = 0
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.net1 = QNetwork(state_len * state_dim, action_dim).to(device)
        self.net2 = QNetwork(state_len * state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(
            list(self.net1.parameters()) + list(self.net2.parameters()), lr=lr
        )

    def take_action(self, state, epsilon=0.01) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            U1, Wait1 = self.net1(state)
            U2, Wait2 = self.net2(state)
            q_values = (
                self.weights[0] * (U1 + U2) / 2 + self.weights[1] * (Wait1 + Wait2) / 2
            )
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(device)  # 转换为 FloatTensor
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)  # 转换为 LongTensor
        rewards = torch.FloatTensor(rewards).to(device)  # 转换为 FloatTensor
        next_states = torch.FloatTensor(next_states).to(device)  # 转换为 FloatTensor
        dones = torch.FloatTensor(dones).to(device)  # 转换为 FloatTensor

        for net in [self.net1, self.net2]:
            U, Wait = net(states)
            current_q_U = U.gather(1, actions).squeeze()
            current_q_Wait = Wait.gather(1, actions).squeeze()

            with torch.no_grad():
                next_U, next_Wait = net(next_states)
                next_q_U = next_U.max(1)[0]
                next_q_Wait = next_Wait.max(1)[0]
                target_U = rewards[:, 0] + self.gamma * next_q_U * (1 - dones)
                target_Wait = rewards[:, 1] + self.gamma * next_q_Wait * (1 - dones)

            loss_U = F.mse_loss(current_q_U, target_U)
            loss_Wait = F.mse_loss(current_q_Wait, target_Wait)
            total_loss = (loss_U + loss_Wait) / 2

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if self.learn_step_counter % self.target_update_interval == 0:
                self.net1.load_state_dict(self.net1.state_dict())
                self.net2.load_state_dict(self.net2.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path: str):
        torch.save(
            {"net1": self.net1.state_dict(), "net2": self.net2.state_dict()}, path
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.net1.load_state_dict(checkpoint["net1"])
        self.net2.load_state_dict(checkpoint["net2"])
        self.net1.eval()
        self.net2.eval()

if __name__ == "__main__":
    # 初始化算法
    algorithms = {
        "DMDDQN": DMDDQN(
            local_state_dim=PARAMS["local_state_dim"],
            local_state_len=PARAMS["local_state_len"],
            action_dim=PARAMS["action_dim"],
            weights=PARAMS["weights"],
            gamma=PARAMS["gamma"],
            lr=PARAMS["actor_lr"],
        ),
        "DDQN": DDQN(
            PARAMS["local_state_dim"],
            PARAMS["local_state_len"],
            PARAMS["action_dim"],
            weights=PARAMS["weights"],
        ),
        "THDQN": THDQN(
            PARAMS["local_state_dim"],
            PARAMS["local_state_len"],
            PARAMS["action_dim"],
            weights=PARAMS["weights"],
        ),
        "SHDQN": SHDQN(
            PARAMS["local_state_dim"],
            PARAMS["local_state_len"],
            PARAMS["action_dim"],
            weights=PARAMS["weights"],
        ),
    }

    # 训练过程
    env = TrainingEnv(
        action_dim=PARAMS["action_dim"],
        machine_num=PARAMS["machine_num"],
        E_ave=PARAMS["E_ave"],
        new_insert=PARAMS["new_insert"],
    )
    num_episodes = PARAMS["num_episodes"]

    for algo_name, agent in algorithms.items():
        for episode in range(num_episodes):
            actions = [0, 0, 0, 0]
            state = env.reset(episode)
            total_reward = 0
            done = False
            while not done:
                action = agent.take_action(state)
                actions[action] += 1
                next_state, reward, done, truncated = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                state = next_state
            print(
                f"{algo_name}:Episode {episode+1}/{num_episodes}, Epsilon: {agent.epsilon:.4f},slack_time {reward[1]:.4f}, makespan: {env.time_step}, actions: {actions}"
            )
        agent.save_model(f"HFSD/RL/{algo_name}.pth")
