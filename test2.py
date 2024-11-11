import numpy as np
import random
import time
import torch
class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        初始化经验回放缓冲区
        :param buffer_size: 缓冲区的最大容量
        :param batch_size: 每次采样时的批量大小
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12):
        """
        将新的经验添加到缓冲区
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 收到的奖励
        :param next_state: 下一状态
        :param done: 是否为终止状态
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        
        # 保存经验到缓冲区
        self.buffer[self.position] = (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12)
        # 更新存储位置
        self.position = (self.position + 1) % self.buffer_size

    def sample(self,batch_size):
        """
        随机采样一个批量的经验
        :return: 经验批次 (states, actions, rewards, next_states, dones)
        """
        # 确保缓冲区内有足够样本
        if len(self.buffer) < batch_size:
            raise ValueError("缓冲区中经验不足，无法进行采样")
        
        batch = random.sample(self.buffer,batch_size)
        s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12 = zip(*batch)
        return (s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12)

    def __len__(self):
        """
        返回当前缓冲区的样本数
        """
        return len(self.buffer)

rb = ReplayBuffer(10000)
add_time = 0
ast = time.time()
for _ in range(10_000):
    s1 = np.random.rand(30,32)
    s2 = np.random.rand(30,32)
    s3 = np.random.rand(1,64)
    s4 = np.random.rand(1,32)
    s5 = np.random.rand(30,32)
    s6 = np.random.rand(1,32)
    s7 = np.random.rand(1,32)
    s8 = np.random.rand(1)
    s9 = np.random.rand(1)
    s10 = np.random.rand(1)
    s11 = np.random.rand(20,32)
    s12 = np.random.rand(20,32)
    dt = time.time()
    rb.add(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12)
    add_time += (time.time()-dt)
all = time.time()-ast
print(add_time,all,add_time/all)

tt = time.time()
devive = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
for _ in range(10000):
    s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12 = rb.sample(512)
    s1 = torch.as_tensor(np.array(s1)).to(devive)
    s2 = torch.as_tensor(np.array(s2)).to(devive)
    s3 = torch.as_tensor(np.array(s3)).to(devive)
    s4 = torch.as_tensor(np.array(s4)).to(devive)
    s5 = torch.as_tensor(np.array(s5)).to(devive)
    s6 = torch.as_tensor(np.array(s6)).to(devive)
    s7 = torch.as_tensor(np.array(s7)).to(devive)
    s9 = torch.as_tensor(np.array(s8)).to(devive)
    s10 = torch.as_tensor(np.array(s10)).to(devive)
    s11 = torch.as_tensor(np.array(s11)).to(devive)
    s12 = torch.as_tensor(np.array(s12)).to(devive)
to_tensor_time = time.time()-tt
print(to_tensor_time)
...
