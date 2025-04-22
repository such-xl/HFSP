from collections import namedtuple
from typing import NamedTuple
import numpy as np
import torch

bufferEntity = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward",
        "done",
    ),
)


class ReplayBuffer:
    def __init__(
        self,
        capacity,
        state_seq_len,
        state_dim,
    ):
        self.pos = 0
        self.buffer_size = capacity
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.entity_size = state_dim * 2 + 1 + 1 + 1
        self.buffer = torch.zeros((capacity, self.entity_size)).to(self.device)
        self.seq_len = state_seq_len
        self.is_full = False
        self.points = [0]

    def add(self, data):
        """

        state,actions,next_state,reward,done

        """
        self.buffer[self.pos] *= 0
        p = 0
        for x in data:
            x = np.array(x).ravel()
            self.buffer[self.pos, p : p + x.size] += self.to_torch(x)
            p += x.size
            if self.size() == 0:
                self.points.append(p)
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == self.buffer_size - 1:
            self.is_full = True

    def sample(self, batch_size):
        samples_idx = np.random.randint(0, self.size(), size=batch_size)
        ten = self.buffer[samples_idx, :]
        p = self.points
        return BufferEntity(
            ten[:, p[0] : p[1]].reshape(batch_size, self.seq_len, -1),  # state
            ten[:, p[1] : p[2]],  # action
            ten[:, p[2] : p[3]].reshape(batch_size, self.seq_len, -1),  # next_state
            ten[:, p[3] : p[4]],  # reward                                 # reward
            ten[:, p[4] : p[5]],  # done                                    # done
        )

    def size(self):
        return self.buffer_size if self.is_full else self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if copy:
                return torch.tensor(array, dtype=torch.float, device=self.device)
            return torch.as_tensor(array, dtype=torch.float, device=self.device)


class BufferEntity(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class PPOBuffer:
    """
    简化版PPO缓冲区，直接使用PyTorch Tensor存储数据以提高读取速度
    仅实现基本的存储和获取功能
    """

    def __init__(
        self,
        obs_dim,
        obs_len,
        state_dim,
        state_len,
        buffer_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化PPO缓冲区

        参数:
            obs,
            obs_mask,
            action,
            reward,
            next_obs,
            done,
            global_state,
            state_mask,
            next_global_state,
            log_prob,
        """
        self.device = device
        self.obs = torch.zeros(
            (buffer_size, obs_len, obs_dim), dtype=torch.float32, device=device
        )
        self.obs_masks = torch.zeros(
            (buffer_size, obs_len), dtype=torch.bool, device=device
        )
        self.actions = torch.zeros((buffer_size), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((buffer_size), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(
            (buffer_size, obs_len, obs_dim), dtype=torch.float32, device=device
        )
        self.next_obs_masks = torch.zeros(
            (buffer_size, obs_len), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros((buffer_size), dtype=torch.bool, device=device)
        self.global_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self.next_global_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self.log_probs = torch.zeros((buffer_size), dtype=torch.float32, device=device)

        self.job = [0 for _ in range(buffer_size)]
        self.ptr = 0
        self.buffer_size = buffer_size
        self.full = False

    def push(
        self,
        obs,
        obs_mask,
        action,
        reward,
        next_obs,
        next_obs_mask,
        done,
        global_state,
        next_global_state,
        log_prob,
        job_id
    ):
        """
        存储一个时间步的交互数据
        支持直接存入tensor或numpy数组
        """

        with torch.no_grad():
            # 将数据存入缓冲区
            self.obs[self.ptr] = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            )
            self.obs_masks[self.ptr] = torch.tensor(
                obs_mask, dtype=torch.bool, device=self.device
            )
            self.actions[self.ptr] = torch.tensor(
                action, dtype=torch.int64, device=self.device
            )
            self.rewards[self.ptr] = torch.tensor(
                reward, dtype=torch.float32, device=self.device
            )
            self.next_obs[self.ptr] = torch.tensor(
                next_obs, dtype=torch.float32, device=self.device
            )
            self.next_obs_masks[self.ptr] = torch.tensor(
                next_obs_mask, dtype=torch.bool, device=self.device
            )
            self.dones[self.ptr] = torch.tensor(
                done, dtype=torch.bool, device=self.device
            )
            self.global_states[self.ptr] = torch.tensor(
                global_state, dtype=torch.float32, device=self.device
            )
            self.next_global_states[self.ptr] = torch.tensor(
                next_global_state, dtype=torch.float32, device=self.device
            )
            self.log_probs[self.ptr] = torch.tensor(
                log_prob, dtype=torch.float32, device=self.device
            )
            self.job[self.ptr] = job_id
            if self.ptr > 0:
                prev_idx = (self.ptr - 1) % self.buffer_size
                self.next_obs[prev_idx] = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                )
                self.next_obs_masks[prev_idx] = torch.tensor(
                    obs_mask, dtype=torch.bool, device=self.device
                )
                self.next_global_states[prev_idx] = torch.tensor(
                    global_state, dtype=torch.float32, device=self.device
                )
                self.dones[prev_idx] = torch.tensor(
                    False, dtype=torch.bool, device=self.device
                )
        # 更新指针
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
            raise ValueError("Buffer is full")

    def get(self):
        """
        获取缓冲区中的所有数据
        返回: 包含所有数据的字典，所有数据都是tensor格式
        """
        if self.full:
            valid_indices = slice(0, self.buffer_size)
        else:
            valid_indices = slice(0, self.ptr)

        data_dict = {
            "obs": self.obs[valid_indices],
            "obs_masks": self.obs_masks[valid_indices],
            "actions": self.actions[valid_indices],
            "rewards": self.rewards[valid_indices],
            "next_obs": self.next_obs[valid_indices],
            "next_obs_mask": self.next_obs_masks[valid_indices],
            "dones": self.dones[valid_indices],
            "global_states": self.global_states[valid_indices],
            "next_global_states": self.next_global_states[valid_indices],
            "log_probs": self.log_probs[valid_indices],
        }

        return data_dict

    def update_last_reward(self, reward):
        self.rewards[(self.ptr - 1) % self.buffer_size] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
    def update_reward(self,reward,jid):
        for i in range(self.ptr):
            if self.job[i] == jid:
                self.rewards[i] = torch.tensor(
                    reward, dtype=torch.float32, device=self.device
                )
    def clear(self):
        """
        清空缓冲区
        """
        self.ptr = 0
        self.full = False

    def __len__(self):
        """
        返回缓冲区中实际存储的经验数量
        """
        if self.full:
            return self.buffer_size
        else:
            return self.ptr
