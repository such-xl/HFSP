from collections import  namedtuple
from typing import NamedTuple
import numpy as np
import torch

bufferEntity = namedtuple('Transition',(
    "state","state_mask","machine_action","action_mask","reward","done","next_state","next_state_mask","next_machine_action"
))
class ReplayBuffer:
    def __init__(self, capacity,state_seq_len,state_dim,machine_action_dim,machine_seq_len):
        self.pos = 0
        self.buffer_size = capacity
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.entity_size = state_seq_len*state_dim*2 + state_seq_len*2 + machine_action_dim*machine_seq_len*2  + 1 + 1 + machine_seq_len*state_seq_len
        self.buffer = torch.zeros((capacity,self.entity_size)).to(self.device)
        self.state_seq_len = state_seq_len
        self.machine_seq_len = machine_seq_len

        self.is_full = False
        self.points = [0]
    def add(self,data):
        """
        
        state,state_mask,machine_action,reward,done,next_state,next_state_mask,next_machine_action,next_action_mask

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
        return BufferEntity(ten[:,p[0]:p[1]].reshape((batch_size,self.state_seq_len,-1)), # state
                            ten[:, p[1]: p[2]],                                             # state_mask
                            ten[:, p[2]: p[3]].reshape((batch_size,self.machine_seq_len,-1)), # machine action
                            ten[:, p[3]: p[4]],                                             # reward
                            ten[:, p[4]: p[5]],                                              # done
                            ten[:, p[5]: p[6]].reshape((batch_size,self.state_seq_len,-1)), # next_state
                            ten[:, p[6]: p[7]],                                              # next_state_mask
                            ten[:, p[7]: p[8]].reshape((batch_size,self.machine_seq_len,-1)), # next machine action
                            ten[:, p[8]: p[9]].reshape((batch_size,self.machine_seq_len,-1))                                              # next action mask
                            )

    def size(self):
        return self.buffer_size if self.is_full else self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if copy:
                return torch.tensor(array, device=self.device, dtype=torch.float)
            return torch.as_tensor(array, device=self.device, dtype=torch.float)


class BufferEntity(NamedTuple):
    states: torch.Tensor
    state_masks:torch.Tensor
    machine_actions:torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor
    next_state_masks: torch.Tensor
    next_machine_actions: torch.Tensor
    next_action_masks: torch.Tensor