import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from params import PARAMS
from scheduling_env.training_env import TrainingEnv



class Actor(nn.Module):
    def __init__(self, state, action_dim, hidden_dim=128):
        super().__init__()
        input_dim = state
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = state.view(state.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

    def sample(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


class Critic(nn.Module):
    def __init__(self, state, action_dim, hidden_dim=128):
        super().__init__()
        input_dim = state + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state = state.view(state.size(0), -1)
        action_onehot = F.one_hot(action, num_classes=4).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, local_s, global_s, a, r, next_local_s, next_global_s, d):
        self.buffer.append((local_s, global_s, a, r, next_local_s, next_global_s, d))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        local_s, global_s, a, r, next_local_s, next_global_s, d = map(
            np.array, zip(*batch)
        )
        r = np.stack(r)  # [batch, 2]
        return (
            torch.FloatTensor(local_s),
            torch.FloatTensor(global_s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(next_local_s),
            torch.FloatTensor(next_global_s),
            torch.FloatTensor(d),
        )


class HMPSAC:
    def __init__(
        self,
        local_state_dim,
        local_state_len,
        global_state_dim,
        global_state_len,
        action_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        a_lr=3e-4,
        c_lr=3e-5,
        weights=[0.5, 0.5],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(local_state_dim*local_state_len, action_dim).to(self.device)
        self.critic = Critic(global_state_dim*global_state_len, action_dim).to(self.device)
        self.critic_target = Critic(global_state_dim*global_state_len, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), a_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), c_lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()
        self.reward_weights = torch.tensor(weights).to(self.device)

    def take_action(self, local_state):
        local_state = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(local_state)
        return action.item()

    def store_transition(self, *args):
        self.replay_buffer.add(*args)

    def update(self, batch_size):
        (local_s, global_s, a, r, next_local_s, next_global_s, d) = (
            self.replay_buffer.sample(batch_size)
        )
        local_s, global_s, a, r, next_local_s, next_global_s, d = (
            local_s.to(self.device),
            global_s.to(self.device),
            a.to(self.device),
            r.to(self.device),
            next_local_s.to(self.device),
            next_global_s.to(self.device),
            d.to(self.device),
        )

        reward = (r @ self.reward_weights).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_local_s)
            target_q1, target_q2 = self.critic_target(next_global_s, next_action)
            target_q = torch.min(
                target_q1, target_q2
            ) - self.alpha * next_log_prob.unsqueeze(1)
            target_value = reward + (1 - d.unsqueeze(1)) * self.gamma * target_q

        current_q1, current_q2 = self.critic(global_s, a)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(
            current_q2, target_value
        )
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, log_prob = self.actor.sample(local_s)
        q1, q2 = self.critic(global_s, new_action)
        actor_loss = (self.alpha * log_prob.unsqueeze(1) - torch.min(q1, q2)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        return actor_loss.item(), critic_loss.item()

    def save_model(self, path):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])


def train(agent, env, num_episodes=300, batch_size=64, min_buffer_size=1000):
    for episode in range(num_episodes):
        actions = [0, 0, 0, 0]
        local_state = env.reset(1)
        global_state = env.get_global_state()
        done, truncated = False, False
        actor_loss, cirtir_loss = 0, 0
        while not done and not truncated:
            action = agent.take_action(local_state)
            actions[action] += 1
            next_local, reward, done, truncated = env.step(action)
            next_global = env.get_global_state()
            local_state, global_state = next_local, next_global
            agent.store_transition(
                local_state, global_state, action, reward, next_local, next_global, done
            )
            if len(agent.replay_buffer.buffer) > min_buffer_size:
                actor_loss, cirtir_loss = agent.update(batch_size)
        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.4f}, cirtir_loss {cirtir_loss:.4f} ,slack_time {reward[1]:.4f} make_span {env.time_step},actions:{actions}"
        )
    agent.save_model(f"HFSD/RL/HMPSAC.pth")


if __name__ == "__main__":
    env = TrainingEnv(
        action_dim=PARAMS["action_dim"],
        machine_num=PARAMS["machine_num"],
        E_ave=PARAMS["E_ave"],
        new_insert=PARAMS["new_insert"],
    )
    agent = HMPSAC(
        local_state_dim=PARAMS["local_state_dim"],
        local_state_len=PARAMS["local_state_len"],
        global_state_dim=PARAMS["global_state_dim"],
        global_state_len=PARAMS["global_state_len"],
        action_dim=PARAMS["action_dim"],
        weights =  PARAMS["weights"],
        gamma=PARAMS["gamma"],
        a_lr=PARAMS["actor_lr"],
        c_lr=PARAMS["critic_lr"],
    )
    train(agent, env)
