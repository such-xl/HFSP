import numpy as np

from .model import Actor
class Agent():
    def __init__(self,epsilon) -> None:
        self.actor = Actor(128,128,5,128,1)
        self.epsilon = epsilon
        pass
    def take_action(self,s_p_m,s_p_j,s_o_j,act_jobs):
        if np.random.random() < self.epsilon:
            action  = np.random.randint(-1,len(act_jobs))
        action  = np.random.randint(-1,len(act_jobs))
        return action
    