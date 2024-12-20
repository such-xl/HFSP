from collections import deque, namedtuple
from typing import NamedTuple

import numpy as np
import torch
#import numba as nb

from .utils import StateNorm

bufferEntity = namedtuple('Transition',
                          ('spms', 'spjs', 'sojs', 'actions', 'rewards', 'nspms', 'nspjs', 'nsojs', 'dones',
                           'mask_spj', 'mask_soj', 'mask_nspj', 'mask_nsoj'))


class ReplayBuffer:
    def __init__(self, capacity, job_dim, job_seq_len, machine_dim, machine_seq_len):
        # self.buffer = np.zeros((capacity, 8773))
        self.state_norm = StateNorm(job_dim, job_seq_len, machine_dim, machine_seq_len)

        self.pos = 0
        self.buffer_size = capacity
        self.full = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        with torch.no_grad():
            self.buffer = torch.zeros((capacity, 5173)).to(self.device)

    def add(self, s_p_m, s_p_j, s_o_j, action, reward, n_s_p_m, n_s_p_j, n_s_o_j, done):
        spj, mask_spj = self.state_norm.job_seq_norm(s_p_j, 0)
        soj, mask_soj = self.state_norm.job_seq_norm(s_o_j, 1)
        nspj, mask_nspj = self.state_norm.job_seq_norm(n_s_p_j, 0)
        nsoj, mask_nsoj = self.state_norm.job_seq_norm(n_s_o_j, 1)
        self.buffer[self.pos % self.buffer_size] *= 0
        self.buffer[self.pos % self.buffer_size, 0: 5] += self.to_torch(s_p_m[0])
        self.buffer[self.pos % self.buffer_size, 5: 1265] += self.to_torch(spj.ravel())
        self.buffer[self.pos % self.buffer_size, 1265: 2525] += self.to_torch(soj.ravel())
        self.buffer[self.pos % self.buffer_size, 2525: 2526] += self.to_torch(action)
        self.buffer[self.pos % self.buffer_size, 2526: 2527] += self.to_torch(reward)
        self.buffer[self.pos % self.buffer_size, 2527: 2532] += self.to_torch(n_s_p_m[0])
        self.buffer[self.pos % self.buffer_size, 2532: 3792] += self.to_torch(nspj.ravel())
        self.buffer[self.pos % self.buffer_size, 3792: 5052] += self.to_torch(nsoj.ravel())
        self.buffer[self.pos % self.buffer_size, 5052: 5053] += self.to_torch((done))
        self.buffer[self.pos % self.buffer_size, 5053: 5083] += self.to_torch(mask_spj)
        self.buffer[self.pos % self.buffer_size, 5083: 5113] += self.to_torch(mask_soj)
        self.buffer[self.pos % self.buffer_size, 5113: 5143] += self.to_torch(mask_nspj)
        self.buffer[self.pos % self.buffer_size, 5143: 5173] += self.to_torch(mask_nsoj)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        samples_idx = np.random.randint(0, self.size(), size=batch_size)
        ten = self.buffer[samples_idx, :]
        # ten = self.to_torch(tmp)
        return BufferEntity(ten[:, 0:5].reshape((batch_size,1,-1)),
                            ten[:, 5: 1265].reshape((batch_size, 30, -1)),
                            ten[:, 1265: 2525].reshape((batch_size, 30, -1)),
                            ten[:, 2525: 2526],
                            ten[:, 2526: 2527], 
                            ten[:, 2527: 2532].reshape((batch_size,1,-1)),
                            ten[:, 2532: 3792].reshape((batch_size, 30, -1)),
                            ten[:, 3792: 5052].reshape((batch_size, 30, -1)),
                            ten[:, 5052: 5053], 
                            ten[:, 5053: 5083], 
                            ten[:, 5083: 5113],
                            ten[:, 5113: 5143], 
                            ten[:, 5143: 5173],
                            )
        # return bufferEntity(*container)
        # return tmp

    def size(self):
        return self.buffer_size if self.full else self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if copy:
                return torch.tensor(array, device=self.device, dtype=torch.float)
            return torch.as_tensor(array, device=self.device, dtype=torch.float)


class BufferEntity(NamedTuple):
    spms: torch.Tensor
    spjs: torch.Tensor
    sojs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    nspms: torch.Tensor
    nspjs: torch.Tensor
    nsojs: torch.Tensor
    dones: torch.Tensor
    mask_spj: torch.Tensor
    mask_soj: torch.Tensor
    mask_nspj: torch.Tensor
    mask_nsoj: torch.Tensor
