from collections import  namedtuple
from typing import NamedTuple
import numpy as np
import torch

bufferEntity = namedtuple('Transition',(
    "state","action","next_state","reward","done",
))
class ReplayBuffer:
    def __init__(self, capacity,
                 state_seq_len,
                        state_dim,
                ):
        self.pos = 0
        self.buffer_size = capacity
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.entity_size = state_dim*2 + 1 + 1 + 1
        self.buffer = torch.zeros((capacity,self.entity_size)).to(self.device)
        self.seq_len = state_seq_len
        self.is_full = False
        self.points = [0]
    def add(self,data):
        """
        
        state,actions,next_state,reward,done

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
        return BufferEntity(ten[:,p[0]:p[1]].reshape(batch_size,self.seq_len,-1),           # state
                            ten[:, p[1]: p[2]],          # action
                            ten[:, p[2]: p[3]].reshape(batch_size,self.seq_len,-1),          # next_state
                            ten[:, p[3]: p[4]],          # reward                                 # reward
                            ten[:, p[4]: p[5]],          #done                                    # done 
                            )

    def size(self):
        return self.buffer_size if self.is_full else self.pos

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if copy:
                return torch.tensor(array,dtype=torch.float,device=self.device)
            return torch.as_tensor(array,dtype=torch.float, device=self.device)


class BufferEntity(NamedTuple):
    states: torch.Tensor
    actions:torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor