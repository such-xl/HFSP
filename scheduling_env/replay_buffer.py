from collections import  namedtuple
from typing import NamedTuple
import numpy as np
import torch

bufferEntity = namedtuple('Transition',(
    "states","machine_states","actions","action_masks","rewards","dones","next_states","next_machine_states","next_action_masks"
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
        
        state,machine_state,actions,action_mask,reward,done,next_state,next_machine_state,next_action_mask

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
        return BufferEntity(ten[:,p[0]:p[1]].reshape((batch_size,self.state_seq_len,-1)),  # state
                            ten[:, p[1]: p[2]].reshape((batch_size,self.machine_seq_len,-1)),  # machine_state
                            ten[:, p[2]: p[3]],                                     # machine action
                            ten[:, p[3]: p[4]].reshape((batch_size,self.machine_seq_len,-1)),                                             # action_mask
                            ten[:, p[4]: p[5]],                                              # reward
                            ten[:, p[5]: p[6]],                                     #done
                            ten[:, p[6]: p[7]].reshape((batch_size,self.state_seq_len,-1)),    # next_state
                            ten[:, p[7]: p[8]].reshape((batch_size,self.machine_seq_len,-1)), # next machine state
                            ten[:, p[8]: p[9]].reshape((batch_size,self.machine_seq_len,-1))   # next action mask
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
    machine_state:torch.Tensor
    actions:torch.Tensor
    action_masks:torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor
    next_machine_states: torch.Tensor
    next_action_masks: torch.Tensor