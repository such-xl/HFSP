import numpy as np
from scheduling_env.fjsp_eval_env import TRAN_ENV
from params import PARAMS
env = TRAN_ENV(
    state_dim=55,
    action_dim=5,
    machine_num=15,
    max_job_num=70,
    lambda_rate=0.05,
    job_file_path="experiment/fjsp/fjsp_same.json",
    seed_list=PARAMS['seed_list'],
)
tardsum = 0
for i in range(100):
    env.reset()
    done = False
    while not done:
        _, _, done,_, info = env.step(np.random.randint(0,5))
        if done:
           tardsum+=sum(info["tardiness"])
print(tardsum)