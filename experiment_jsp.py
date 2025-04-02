import os
import torch
import numpy as np
import natsort
import json

from scheduling_env.eval_env import FJSEvalEnv
from scheduling_env.MAPPO import AsyncMAPPO
from scheduling_env.basic_scheduling_algorithms import EDD,MS,SRO,CR

def eval_mappo(env: FJSEvalEnv,mappo: AsyncMAPPO):
        
    obs_i, obs_mask = env.reset()
    done, truncated = False, False
    while not done and not truncated:

        action, _, _ = mappo.select_action(obs_i, obs_mask, eval_mode=True)

        next_obs,next_obs_mask,_,done,truncated= env.step(action)
        obs_i = next_obs
        obs_mask = next_obs_mask
    tards_record = [job.tard_time for job in env.job_list]
    return tards_record
         
def eval_sr(env,sr_type="CR"):
    _,_ = env.reset()
    done,truncated = False,False
    while not done and not truncated:
        if sr_type == "EDD":
            action = EDD(env.available_jobs)
        elif sr_type=="SRO":
            action = SRO(env.available_jobs,env.time_step)
        elif sr_type == "MS":
            action = MS(env.available_jobs,env.time_step)
        elif sr_type == "CR":
            action = CR(env.available_jobs,env.time_step)
        reward, done, truncated = env.step_by_sr(action)
    
    tards_record = [job.tard_time for job in env.job_list]
    return tards_record
PARAMS = {
    "num_episodes": 1,
    "batch_size": 24,
    "actor_lr": 6e-5,
    "critic_lr": 4e-4,
    "gamma": 1,
    "obs_dim": 6,
    "obs_len": 5,
    "global_state_dim": 6,
    "global_state_len": 30,
    "action_dim": 4,
    "max_machine_num": 10,
    "max_job_num": 28,
    "share_parameters": False,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__))
    + "/experiment/jsp/job_data/",
    "train": False,
    "idle_action": False,
    "model_path": "models/jsp.pth",
    "UR": [70,80,90],
    "SR":["EDD","MS","SRO","CR"]
}
if __name__ == "__main__":


    mappo = AsyncMAPPO(
        n_agents=PARAMS["max_machine_num"],
        obs_dim=PARAMS["obs_dim"],
        obs_len=PARAMS["obs_len"],
        global_state_dim=PARAMS["global_state_dim"],
        global_state_len=PARAMS["global_state_len"],
        act_dim=PARAMS["action_dim"],
        actor_lr=PARAMS["actor_lr"],
        critic_lr=PARAMS["critic_lr"],
        gamma=PARAMS["gamma"],
        num_heads=PARAMS["num_heads"],
        device=PARAMS["device"],
        model_save_path=PARAMS["model_path"],
    )
    mappo.load_model()
    tards_record = {}
    for ur in PARAMS["UR"]:
        tards_record[ur] = {
            "RL":[],
            "EDD":[],
            "MS":[],
            "SRO":[],
            "CR":[],
        }

    for ur in PARAMS["UR"]:
        data_path = PARAMS["data_path"] + str(ur) + "/"
        jobs_name = natsort.natsorted(os.listdir(data_path))
        for job in jobs_name:     
            env_rl = FJSEvalEnv(
                obs_dim=PARAMS["obs_dim"],
                obs_len=PARAMS["obs_len"],
                action_dim=PARAMS["action_dim"],
                max_job_num=PARAMS["max_job_num"],
                file_path=data_path + job,
                )
            record = eval_mappo(env_rl,mappo)
            print(f"UR:{ur} {job} total tard:{np.sum(record)}")
            # record = eval_sr(env)
            tards_record[ur]["RL"].append(record)
    

    
    for ur in PARAMS["UR"]:
        data_path = PARAMS["data_path"] + str(ur) + "/"
        jobs_name = natsort.natsorted(os.listdir(data_path))
        for sr in PARAMS["SR"]:
            for job in jobs_name:
                env_sr = FJSEvalEnv(
                obs_dim=PARAMS["obs_dim"],
                obs_len=PARAMS["obs_len"],
                action_dim=PARAMS["action_dim"],
                max_job_num=PARAMS["max_job_num"],
                file_path=data_path + job,
                )
                record = eval_sr(env_sr,sr)
                print(f"SR:{sr} UR:{ur} {job} total tard:{np.sum(record)}")
                tards_record[ur][sr].append(record)
            
    with open("experiment/jsp/my_resurt.json", "w") as f:
        json.dump(tards_record, f)