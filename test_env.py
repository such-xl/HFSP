import copy
import numpy as np
import json
from params import PARAMS
from scheduling_env.training_env import TrainingEnv
from scheduling_env.Instance_Generator import Instance_Generator
from scheduling_env.model import PPO
np.random.seed(42)
EVAL_EPISODE = 10

METHODS = ["RL", "SPT", "LPT", "LRPT","SRPT","Random"]

for episode in range(10):
    # 只生成一次
    jobs_list, arrivals_list = Instance_Generator(
        M_num=PARAMS["machine_num"],
        E_ave=PARAMS["E_ave"],
        New_insert=PARAMS["new_insert"]
    )
    # 为每个方法都用这同一批 jobs
    envs = [
        TrainingEnv(
            action_dim=PARAMS["action_dim"],
            machine_num=PARAMS["machine_num"],
            E_ave=PARAMS["E_ave"],
            new_insert=PARAMS['new_insert'],
            job_info_list=jobs_list,
            job_arrival_time=arrivals_list
        )
        for _ in METHODS
    ]

envs    