import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from .replay_buffer import PPOBuffer
from .network import ActorNetwork, CriticNetwork


# 设置随机种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


class MAPPOAgent:
    def __init__(
        self,
        agent_id,
        obs_dim,
        obs_len,
        act_dim,
        global_state_dim,
        global_state_len,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        num_heads=5,
        device=torch.device("cpu"),
        model_save_path="models/main.pth",
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
        self.device = device
        self.model_save_path = model_save_path
        # 创建策略网络和价值网络
        self.actor = ActorNetwork(obs_dim, act_dim, num_heads).to(device)
        # 价值网络使用全局状态作为输入
        self.critic = CriticNetwork(global_state_dim, num_heads).to(device)
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 旧策略，用于计算重要性采样比率
        self.old_actor = ActorNetwork(obs_dim, act_dim, num_heads).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

    def select_action(self, obs, obs_mask, tau, hard,eval_mode=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        obs_mask = torch.BoolTensor(obs_mask).to(self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy = self.actor.get_action(
                obs_tensor, obs_mask, tau, hard,eval_mode=eval_mode
            )

        return action, log_prob.cpu().numpy(), entropy.cpu().numpy()

    def evaluate_state(self, global_state, global_state_mask=None):
        state_tensor = torch.FloatTensor(global_state).to(self.device)
        state_mask = torch.BoolTensor(global_state_mask).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor, state_mask)
        return value.cpu().numpy()

    def store_experience(
        self,
        obs,
        obs_mask,
        action,
        reward,
        next_obs,
        next_obs_mask,
        done,
        global_state,
        state_mask,
        next_global_state,
        next_state_mask,
        log_prob,
        buffer,
    ):
        buffer.push(
            obs,
            obs_mask,
            action,
            reward,
            next_obs,
            next_obs_mask,
            done,
            global_state,
            state_mask,
            next_global_state,
            next_state_mask,
            log_prob,
        )

    def update_old_policy(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
    def update_last_reward(self,reward,buffer):
        buffer.update_last_reward(reward)
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards, dtype=torch.float32,device=self.device)
        returns = torch.zeros_like(rewards, dtype=torch.float32,device=self.device)
        gae = torch.tensor(0.0, dtype=torch.float32,device=self.device)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].to(torch.float32)) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].to(torch.float32)) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns

    def update(self, buffer, batch_size=8, epochs=10,tau=0.1,hard=False):
        if len(buffer) < batch_size:
            return 0, 0, 0  # 如果没有足够的样本，则不更新

        # 获取所有经验
        batch = buffer.get()
        obs_tensor = batch["obs"]
        obs_mask_tensor = batch["obs_masks"]
        action_tensor = batch["actions"]
        reward_tensor = batch["rewards"]
        next_obs_tensor = batch["next_obs"]
        next_obs_mask_tensor = batch["next_obs_mask"]
        done_tensor = batch["dones"]
        global_state_tensor = batch["global_states"]
        state_mask_tensor = batch["state_masks"]
        next_global_state_tensor = batch["next_global_states"]
        next_state_mask_tensor = batch["next_state_masks"]
        old_log_prob_tensor = batch["log_probs"]
        # 计算当前价值估计
        with torch.no_grad():
            values = (
                self.critic(global_state_tensor, state_mask_tensor)
                .squeeze(-1)
            )
            next_values = (
                self.critic(next_global_state_tensor, next_state_mask_tensor)
                .squeeze(-1)
            )

        # 计算优势函数和回报
        advantages_tensor, returns_tensor = self.compute_advantages(
            reward_tensor, values, next_values, done_tensor
        )

        # 将优势和回报转换为张量
        # advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        # returns_tensor = torch.FloatTensor(returns).to(self.device)

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
            rand_idx = torch.randperm(len(batch)).to(self.device)
            n_batches = len(rand_idx) // batch_size + (
                1 if len(rand_idx) % batch_size > 0 else 0
            )

            for i in range(n_batches):
                batch_idx = rand_idx[i * batch_size : (i + 1) * batch_size]

                # 提取小批量数据
                batch_obs = obs_tensor[batch_idx]
                batch_obs_mask = obs_mask_tensor[batch_idx]
                batch_action = action_tensor[batch_idx]
                batch_old_log_prob = old_log_prob_tensor[batch_idx]
                batch_advantage = advantages_tensor[batch_idx]
                batch_return = returns_tensor[batch_idx]
                batch_global_state = global_state_tensor[batch_idx]
                batch_state_mask = state_mask_tensor[batch_idx]

                # 计算当前策略的动作概率
                logits = self.actor(batch_obs, batch_obs_mask)  # 获取logits
                # # logits = logits.masked_fill(batch_obs_mask[:,:-1], float("-inf"))
                # probs = F.softmax(logits, dim=-1)  # 显式应用softmax
                # dist = Categorical(probs)
                # curr_log_prob = dist.log_prob(batch_action)
                # entropy = dist.entropy().mean()
                # 计算当前策略的动作概率
                            # 软或硬Gumbel-Softmax采样
                if hard:  # 如果硬 Gumbel-Softmax
                    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
                    gumbel_logits = (logits + gumbels) / tau
                    index = gumbel_logits.argmax(dim=-1)
                    y_hard = torch.zeros_like(logits).scatter_(-1, index.unsqueeze(-1), 1.0)
                    y = F.softmax(gumbel_logits, dim=-1)
                    y_out = y_hard - y.detach() + y
                else:  # 软 Gumbel-Softmax
                    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
                    y_out = F.softmax((logits + gumbels) / tau, dim=-1)

                            # 计算当前策略的动作概率分布
                dist = Categorical(y_out)
                curr_log_prob = dist.log_prob(batch_action)  # 当前采样的对数概率
                entropy = dist.entropy().mean()  # 熵值
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
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        # 更新旧策略
        self.update_old_policy()

        # 清空缓冲区
        buffer.clear()

        # 计算平均损失
        n_updates = epochs * n_batches
        return (
            actor_loss_epoch / n_updates,
            critic_loss_epoch / n_updates,
            entropy_epoch / n_updates,
        )
    def save(self):
        print(self.model_save_path)
        torch.save(self.actor.state_dict(),self.model_save_path)
    def load(self):
        state_dict = torch.load(self.model_save_path)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
class AsyncMAPPO:
    def __init__(
        self,
        n_agents,
        obs_dim,
        obs_len,
        global_state_dim,
        global_state_len,
        act_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_capacity=10000,
        gae_lambda=0.95,
        num_heads=5,
        buffer_size=10000,
        device=torch.device("cpu"),
        model_save_path="models/main.pth",
    ):
        # agents with share networt but replay buffer
        self.agents = MAPPOAgent(
            agent_id=0,
            obs_dim=obs_dim,
            obs_len=obs_len,
            act_dim=act_dim,
            global_state_dim=global_state_dim,
            global_state_len=global_state_len,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            eps_clip=eps_clip,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            gae_lambda=gae_lambda,
            num_heads=num_heads,
            device=device,
            model_save_path=model_save_path,
        )
        self.buffers = [
            PPOBuffer(
                obs_dim=obs_dim,
                obs_len=obs_len,
                state_dim=global_state_dim,
                state_len=global_state_len,
                buffer_size=buffer_size,
                device=device,
            )
            for _ in range(n_agents)
        ]
        self.n_agents = n_agents
        self.device = device

    def select_action(self, obs, obs_mask, tau=0.1, hard=False,eval_mode=False):
        """为当前活动智能体选择动作"""
        return self.agents.select_action(obs, obs_mask, tau, hard,eval_mode)

    def update(self, batch_size=64, epochs=100,tau=0.1,hard=False):
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for buffer in self.buffers:
            actor_loss, critic_loss, entropy = self.agents.update(
                buffer, batch_size, epochs,tau=0.1,hard=False
            )
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy

        return (
            total_actor_loss / self.n_agents,
            total_critic_loss / self.n_agents,
            total_entropy / self.n_agents,
        )

    def store_experience(
        self,
        obs,
        obs_mask,
        action,
        reward,
        next_obs,
        next_obs_mask,
        done,
        global_state,
        state_mask,
        next_global_state,
        next_state_mask,
        log_prob,
        agent_id,
    ):
        """存储当前活动智能体的经验"""
        self.agents.store_experience(
            obs,
            obs_mask,
            action,
            reward,
            next_obs,
            next_obs_mask,
            done,
            global_state,
            state_mask,
            next_global_state,
            next_state_mask,
            log_prob,
            self.buffers[agent_id],
        )
    def update_last_reward(self,rewards):
        """更新最后一步的奖励"""
        for i in range(len(rewards)):
            self.agents.update_last_reward(rewards[i],self.buffers[i])

    def evaluate_state(self, global_state, global_state_mask):
        """评估当前状态的价值（从指定智能体的角度）"""
        return self.agents.evaluate_state(global_state, global_state_mask)
    def save_model(self):
        """保存模型"""
        self.agents.save()
    def load_model(self):
        """加载模型"""
        self.agents.load()