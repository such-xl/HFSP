import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random


# 设置随机种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义Actor网络 - 策略网络
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2 * input_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x):
        return F.softmax(self.linear(x), dim=-1)

    def get_action(self, state, available_actions=None):
        probs = self.forward(state)
        # 如果提供了可用动作掩码，则应用它
        if available_actions is not None:
            # 将不可用动作的概率设为0
            probs = probs * available_actions
            # 重新归一化概率
            probs = probs / (probs.sum() + 1e-10)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, dist.entropy()


# 定义Critic网络 - 价值网络
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(CriticNetwork, self).__init__()
        # 状态嵌入层
        self.state_embedding = nn.Linear(input_dim, 2 * input_dim)
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            2 * input_dim, num_heads, batch_first=True
        )
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 1),  # 输出单个值
        )

    def forward(self, x, mask=None):
        x = self.state_embedding(x)
        x = self.attention(x, x, x, key_padding_mask=mask)[0]
        x = x[:, -1, :]
        return self.mlp(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        if len(self.buffer) > 0:
            # 将上一个经验的下一个状态设置为当前状态
            self.buffer[-1][3] = experience[0]
            self.buffer[-1][7] = experience[5]
            self.buffer[-1][8] = experience[6]
            self.buffer[-1][4] = False
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def update_reward(self, reward):
        if len(self.buffer) > 0:
            self.buffer[-1][2] = reward

    def __len__(self):
        return len(self.buffer)


# MAPPO智能体
class MAPPOAgent:
    def __init__(
        self,
        agent_id,
        obs_dim,
        obs_len,
        act_dim,
        global_state_dim,
        global_state_len,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_capacity=10000,
        gae_lambda=0.95,
        num_heads=5,
        device=torch.device("cpu"),
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.act_dim = act_dim
        self.global_state_dim = global_state_dim
        self.global_state_len = global_state_len
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        # 创建策略网络和价值网络
        self.actor = ActorNetwork(obs_dim * obs_len, act_dim).to(device)
        # 价值网络使用全局状态作为输入
        self.critic = CriticNetwork(global_state_dim, num_heads).to(device)

        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)

        # 旧策略，用于计算重要性采样比率
        self.old_actor = ActorNetwork(obs_dim * obs_len, act_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, obs, available_actions=None):
        obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)
        # 将可用动作转换为张量(如果提供了)
        if available_actions is not None:
            available_actions = torch.FloatTensor(available_actions).to(device)

        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(
                obs_tensor, available_actions
            )

        return action, log_prob.cpu().numpy(), entropy.cpu().numpy()

    def evaluate_state(self, global_state, global_state_mask=None):
        state_tensor = torch.FloatTensor(global_state).to(device)
        state_mask = torch.BoolTensor(global_state_mask).to(device)
        with torch.no_grad():
            value = self.critic(state_tensor, state_mask)
        return value.cpu().numpy()

    def store_experience(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        global_state,
        state_mask,
        next_global_state,
        next_state_mask,
        log_prob,
    ):
        self.buffer.push(
            [
                obs,
                action,
                reward,
                next_obs,
                done,
                global_state,
                state_mask,
                next_global_state,
                next_state_mask,
                log_prob,
            ]
        )

    def update_old_policy(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self, batch_size=64, epochs=100):
        if len(self.buffer) < batch_size:
            return 0, 0, 0  # 如果没有足够的样本，则不更新

        # 获取所有经验
        batch = self.buffer.buffer

        # 提取经验数据
        obs_batch = np.array([x[0] for x in batch])
        action_batch = np.array([x[1] for x in batch])
        reward_batch = np.array([x[2] for x in batch])
        next_obs_batch = np.array([x[3] for x in batch])
        done_batch = np.array([x[4] for x in batch])
        global_state_batch = np.array([x[5] for x in batch])
        state_mask_batch = np.array([x[6] for x in batch])
        next_global_state_batch = np.array([x[7] for x in batch])
        next_state_mask_batch = np.array([x[8] for x in batch])
        old_log_prob_batch = np.array([x[9] for x in batch])

        # 转换为张量
        obs_tensor = torch.FloatTensor(obs_batch).to(device)
        action_tensor = torch.LongTensor(action_batch).to(device)
        reward_tensor = torch.FloatTensor(reward_batch).to(device)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(device)
        done_tensor = torch.FloatTensor(done_batch).to(device)
        global_state_tensor = torch.FloatTensor(global_state_batch).to(device)
        state_mask_tensor = torch.BoolTensor(state_mask_batch).to(device)
        next_global_state_tensor = torch.FloatTensor(next_global_state_batch).to(device)
        next_state_mask_tensor = torch.BoolTensor(next_state_mask_batch).to(device)
        old_log_prob_tensor = torch.FloatTensor(old_log_prob_batch).to(device)

        # 计算当前价值估计
        with torch.no_grad():
            values = (
                self.critic(global_state_tensor, state_mask_tensor)
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            next_values = (
                self.critic(next_global_state_tensor, next_state_mask_tensor)
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        # 计算优势函数和回报
        advantages, returns = self.compute_advantages(
            reward_batch, values, next_values, done_batch
        )

        # 将优势和回报转换为张量
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)

        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # 多次迭代更新策略和价值网络
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0

        for _ in range(epochs):
            # 随机采样小批量
            rand_idx = np.random.permutation(len(batch))
            n_batches = len(rand_idx) // batch_size + (
                1 if len(rand_idx) % batch_size > 0 else 0
            )

            for i in range(n_batches):
                batch_idx = rand_idx[i * batch_size : (i + 1) * batch_size]

                # 提取小批量数据
                batch_obs = obs_tensor[batch_idx]
                batch_action = action_tensor[batch_idx]
                batch_old_log_prob = old_log_prob_tensor[batch_idx]
                batch_advantage = advantages_tensor[batch_idx]
                batch_return = returns_tensor[batch_idx]
                batch_global_state = global_state_tensor[batch_idx]
                batch_state_mask = state_mask_tensor[batch_idx]
                # 计算当前策略的动作概率
                probs = self.actor(batch_obs)
                dist = Categorical(probs)
                curr_log_prob = dist.log_prob(batch_action)
                entropy = dist.entropy().mean()

                # 计算重要性采样比率
                ratio = torch.exp(curr_log_prob - batch_old_log_prob)

                # 计算裁剪的目标函数
                surr1 = ratio * batch_advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                    * batch_advantage
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                values = self.critic(batch_global_state, batch_state_mask).squeeze(-1)
                critic_loss = F.mse_loss(values, batch_return)

                # 添加熵正则化
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        # 更新旧策略
        self.update_old_policy()

        # 清空缓冲区
        self.buffer = ReplayBuffer(self.buffer.buffer.maxlen)

        # 计算平均损失
        n_updates = epochs * n_batches
        return (
            actor_loss_epoch / n_updates,
            critic_loss_epoch / n_updates,
            entropy_epoch / n_updates,
        )


# MAPPO框架，管理多个智能体
class MAPPO:
    def __init__(
        self,
        n_agents,
        obs_dim,
        obs_len,
        global_state_dim,
        global_state_len,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_capacity=10000,
        gae_lambda=0.95,
        share_parameters=False,
        num_heads=5,
        device=torch.device("cpu"),
    ):
        self.n_agents = n_agents

        if share_parameters:
            # 所有智能体共享相同的网络参数

            agent = MAPPOAgent(
                0,
                obs_dim,
                obs_len,
                act_dim,
                global_state_dim,
                global_state_len,
                lr,
                gamma,
                eps_clip,
                value_coef,
                entropy_coef,
                buffer_capacity,
                gae_lambda,
                num_heads=num_heads,
                device=device,
            )
            self.agents = [agent] * n_agents
        else:
            # 每个智能体有独立的网络参数
            self.agents = [
                MAPPOAgent(
                    i,
                    obs_dim,
                    obs_len,
                    act_dim,
                    global_state_dim,
                    global_state_len,
                    lr,
                    gamma,
                    eps_clip,
                    value_coef,
                    entropy_coef,
                    buffer_capacity,
                    gae_lambda,
                    num_heads,
                    device,
                )
                for i in range(n_agents)
            ]

    # 用于同步环境的动作选择
    def select_actions(self, obs_list, available_actions_list=None):
        actions = []
        log_probs = []
        entropies = []

        for i, agent in enumerate(self.agents):
            if available_actions_list is not None:
                action, log_prob, entropy = agent.select_action(
                    obs_list[i], available_actions_list[i]
                )
            else:
                action, log_prob, entropy = agent.select_action(obs_list[i])

            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

        return actions, log_probs, entropies

    def store_experiences(
        self,
        obs_list,
        action_list,
        reward_list,
        next_obs_list,
        done_list,
        global_state,
        next_global_state,
        log_prob_list,
    ):
        for i, agent in enumerate(self.agents):
            agent.store_experience(
                obs_list[i],
                action_list[i],
                reward_list[i],
                next_obs_list[i],
                done_list[i],
                global_state,
                next_global_state,
                log_prob_list[i],
            )

    def update_reward(self, reward):
        for agent in self.agents:
            agent.buffer.update_reward(reward)

    def update(self, batch_size=64, epochs=100):
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for agent in self.agents:
            actor_loss, critic_loss, entropy = agent.update(batch_size, epochs)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy

        return (
            total_actor_loss / self.n_agents,
            total_critic_loss / self.n_agents,
            total_entropy / self.n_agents,
        )


# 用于异步决策环境的MAPPO（如Hanabi）
class AsyncMAPPO(MAPPO):
    def __init__(
        self,
        n_agents,
        obs_dim,
        obs_len,
        global_state_dim,
        global_state_len,
        act_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_capacity=10000,
        gae_lambda=0.95,
        share_parameters=False,
        num_heads=5,
        device=torch.device("cpu"),
    ):

        super().__init__(
            n_agents,
            obs_dim,
            obs_len,
            global_state_dim,
            global_state_len,
            act_dim,
            lr,
            gamma,
            eps_clip,
            value_coef,
            entropy_coef,
            buffer_capacity,
            gae_lambda,
            share_parameters,
            num_heads,
            device,
        )

    def select_action(self, obs, agent_id, available_actions=None):
        """为当前活动智能体选择动作"""
        return self.agents[agent_id].select_action(obs, available_actions)

    def store_experience(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        global_state,
        state_mask,
        next_global_state,
        next_state_mask,
        log_prob,
        agent_id,
    ):
        """存储当前活动智能体的经验"""
        self.agents[agent_id].store_experience(
            obs,
            action,
            reward,
            next_obs,
            done,
            global_state,
            state_mask,
            next_global_state,
            next_state_mask,
            log_prob,
        )

    def evaluate_state(self, global_state, global_state_mask, agent_id):
        """评估当前状态的价值（从指定智能体的角度）"""
        return self.agents[agent_id].evaluate_state(global_state, global_state_mask)
