import random
from collections import deque
import numpy as np

from .utils import StateNorm

class ReplayBuffer:
    def __init__(self, capacity,job_dim,job_seq_len,machine_dim,machine_seq_len):
        self.buffer = deque(maxlen=capacity)  # 使用双端队列存储经验，自动丢弃最旧的数据
        self.state_norm = StateNorm(job_dim,job_seq_len,machine_dim,machine_seq_len)

    def add(self,s_p_m,s_p_j,s_o_j, action, reward, n_s_p_m,n_s_p_j,n_s_o_j, done):
        self.buffer.append((s_p_m,s_p_j,s_o_j, action, reward, n_s_p_m,n_s_p_j,n_s_o_j, done))

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)  # 确保不会超过buffer的大小
        samples = random.sample(self.buffer, sample_size)  # 随机抽样

        # 解包样本元组为各自的组件，并转换成numpy数组以便进行批量处理
        s_p_ms,s_p_js,s_o_js, actions, rewards,n_s_p_ms,n_s_p_js,n_s_o_js, dones = zip(*samples)
        s_p_js,mask_spj = self.state_norm.job_seq_norm(s_p_js,0)
        s_o_js,mask_soj = self.state_norm.job_seq_norm(s_o_js,1)
        n_s_p_js,mask_nspj = self.state_norm.job_seq_norm(n_s_p_js,0)
        n_s_o_js,mask_nsoj = self.state_norm.job_seq_norm(n_s_o_js,1)
        return (
            np.array(s_p_ms),
            s_p_js,
            s_o_js,
            actions, 
            np.array(rewards), 
            np.array(n_s_p_ms),
            n_s_p_js,
            n_s_o_js, 
            np.array(dones),
            mask_spj,
            mask_soj,
            mask_nspj,
            mask_nsoj
        )

    def size(self):
        return len(self.buffer)

# #使用示例
# replay_buffer = ReplayBuffer(capacity=10000)
# for _ in range(50):  # 假设添加50个样本
#     s_o_j = np.random.random(4)
#     s_p_j = np.random.random(4)
#     s_p_m = np.random.random(1)
#     action = random.randint(0, 2)  # 假设action是0, 1, 2中的一个
#     reward = random.random()  # 随机奖励
#     next_state = np.random.random(4)
#     done = random.choice([True, False])
#     replay_buffer.add(s_o_j,s_p_j,s_p_m, action, reward, next_state, done)

# # 随机抽样一个大小为10的batch
# batch = replay_buffer.sample(10)
# for i in batch:
#     print(i.shape)