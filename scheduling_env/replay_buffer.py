from collections import  namedtuple
from typing import NamedTuple
import numpy as np
import torch

bufferEntity = namedtuple('Transition',('machine_state','job_state','machine_mask','job_mask','action','next_machine_state','next_job_state','next_machine_mask','next_job_mask','reward','done'))

class ReplayBuffer:
    def __init__(self, capacity, job_dim, job_seq_len, machine_dim, machine_seq_len,action_dim):
        self.pos = 0
        self.buffer_size = capacity
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.entity_size = (job_dim*job_seq_len+machine_dim*machine_seq_len+machine_seq_len+job_seq_len+action_dim)*2 + 1+ 1+1

        self.buffer = torch.zeros((capacity,self.entity_size)).to(self.device)
        self.job_dim = job_dim
        self.job_seq_len = job_seq_len
        self.machine_dim = machine_dim
        self.machine_seq_len = machine_seq_len
        self.action_dim = action_dim
        self.is_full = False
        self.points = [0]
    def add(self,data):
        """
        
        state,state_mask,action,action_mask,next_state,next_action_mask,next_sate_mask,reward,done

        """
        self.buffer[self.pos] *= 0
        p = 0
        for x in data:
            x = np.array(x).ravel()
            self.buffer[self.pos,p:p+x.size] += self.to_torch(x)
            p += x.size
            if self.size()==0:
                self.points.append(p)
        self.pos = (self.pos+1) % self.buffer_size
        if self.pos == self.buffer_size-1:
            self.is_full = True
    def sample(self, batch_size):
        samples_idx = np.random.randint(0, self.size(), size=batch_size)
        ten = self.buffer[samples_idx, :]
        p = self.points
        return BufferEntity(ten[:,p[0]:p[1]].reshape((batch_size,self.machine_seq_len,-1)), # machine state
                            ten[:, p[1]: p[2]].reshape((batch_size,self.job_seq_len, -1)),   # job state
                            ten[:, p[2]: p[3]],                                            # machine mask
                            ten[:, p[3]: p[4]],                                              # job mask
                            ten[:, p[4]: p[5]],                                      # action_mask
                            ten[:, p[5]: p[6]].reshape((batch_size,-1)),                   # action
                            ten[:, p[6]: p[7]].reshape((batch_size,self.machine_seq_len,-1)), # next_machine_state
                            ten[:, p[7]: p[8]].reshape((batch_size,self.job_seq_len, -1)),   # next_job_state
                            ten[:, p[8]: p[9]],                                             # next_machine_mask
                            ten[:, p[9]: p[10]],                                              # next_job_mask
                            ten[:, p[10]: p[11]],                                             # next_action_mask
                            ten[:, p[11]:p[12]],                                             # reward
                            ten[:, p[12]:p[13]],                                            # done 
                            )

    def size(self):
        return self.buffer_size if self.is_full else self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if copy:
                return torch.tensor(array, device=self.device, dtype=torch.float)
            return torch.as_tensor(array, device=self.device, dtype=torch.float)


class BufferEntity(NamedTuple):
    machine_states: torch.Tensor
    job_states:torch.Tensor
    machine_masks:torch.Tensor
    job_masks:torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    next_machine_states: torch.Tensor
    next_job_states: torch.Tensor
    next_machine_masks: torch.Tensor
    next_job_masks: torch.Tensor
    next_action_masks: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor