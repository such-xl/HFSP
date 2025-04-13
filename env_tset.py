import numpy as np
import copy
from scheduling_env.fjsp_eval_env import FJSP_EVAL_ENV
from scheduling_env.basic_scheduling_algorithms import noname_2,EDD
rng = np.random.default_rng(42)
env_1 = FJSP_EVAL_ENV(
    obs_dim=6,
    obs_len=10,
    state_dim=6,
    state_len=10,
    action_dim=10,
    max_job_num=20,
    job_file_path="./fjsp_same.json",
    rng = rng
)

env_2 = copy.deepcopy(env_1)
env_3 = copy.deepcopy(env_1)
for i in range(3):
    obs_i,obs_mask,global_state = env_1.reset()
    done,truncated = False,False
    while not done:
        action = noname_2(env_1.available_jobs,env_1.current_machine,env_1.compute_UR())
        _,done,truncated= env_1.step_by_sr(action)
    print(env_1.time_step,end=" ")

for i in range(3):
    obs_i,obs_mask,global_state = env_2.reset()
    done,tructed = False,False
    while not done:
        action = noname_2(env_2.available_jobs,env_2.current_machine,env_2.compute_UR())
        _,done,truncated= env_2.step_by_sr(action)
    print(env_2.time_step,end=" ")
for i in range(3):
    obs_i,obs_mask,global_state = env_3.reset()
    done,tructed = False,False
    while not done:
        action = noname_2(env_3.available_jobs,env_3.current_machine,env_3.compute_UR())
        _,done,truncated= env_3.step_by_sr(action)
    print(env_3.time_step,end=" ")

