
import numpy as np
from scheduling_env.basic_scheduling_algorithms import PDR_RULES
from scheduling_env.training_env import TrainingEnv
from params import PARAMS

for pdr in range(5):
    tardiness = []
    env = TrainingEnv(
    state_dim=PARAMS["state_dim"],
    action_dim=PARAMS["action_dim"],
    machine_num=PARAMS["machine_num"],
    max_job_num=PARAMS["max_job_num"][0],
    lambda_rate=PARAMS["lambda_rate"][0],
    job_file_path=PARAMS["fjsp_same_path"],
    seed_list=PARAMS["seed_list"],
    )
    for eps in range(5):
        obs = env.reset()
        done = False
        while not done:
            _, _, done,_, info = env.step_by_sr(pdr) 
            if done:
                tardiness.append(info["tardiness"]) 
    print(np.mean(tardiness))