import os
import torch
import numpy as np
import natsort
import json

from scheduling_env.eval_env import FJSEvalEnv
from scheduling_env.MAPPO import AsyncMAPPO
from scheduling_env.basic_scheduling_algorithms import EDD,MS,SRO,CR,noname_2

def eval_mappo(env: FJSEvalEnv,mappo: AsyncMAPPO):
        
    obs_i, obs_mask = env.reset()
    done, truncated = False, False
    actions = [0 for _ in range(4)]
    while not done and not truncated:

        action, _, _ = mappo.select_action(obs_i, obs_mask, eval_mode=True)
        actions[action] += 1
        next_obs,next_obs_mask,_,done,truncated= env.step(action)
        obs_i = next_obs
        obs_mask = next_obs_mask
    tards_record = [job.tard_time for job in env.job_list]
    ur_record = [machine.get_utilization_rate(env.time_step) for machine in env.machines]
    return tards_record,ur_record,actions
         
def eval_sr(env:FJSEvalEnv,sr_type="CR"):
    _,_ = env.reset()
    done,truncated = False,False
    while not done and not truncated:
        if sr_type == "EDD":
            action = EDD(env.available_jobs)[0]
        elif sr_type=="SRO":
            action = SRO(env.available_jobs,env.time_step)[0]
        elif sr_type == "MS":
            action = MS(env.available_jobs,env.time_step)[0]
        elif sr_type == "CR":
            action = CR(env.available_jobs,env.time_step)[0]
        elif sr_type == "noname_2":
            action = noname_2(env.available_jobs,env.current_machine,env.compute_UR())
        reward, done, truncated = env.step_by_sr(action)
    
    tards_record = [job.tard_time for job in env.job_list]
    ur_record = [machine.get_utilization_rate(env.time_step) for machine in env.machines]
    return tards_record,ur_record
PARAMS = {
    "num_episodes": 1,
    "batch_size": 24,
    "actor_lr": 6e-5,
    "critic_lr": 4e-4,
    "gamma": 1,
    "obs_dim": 8,
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
    "model_path": "models/fjsp_same.pth",
    "UR": [70,80,90],
    "SR":["EDD","MS","SRO","CR","noname_2"],
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
    tards_records = {}
    ur_records = {}
    for ur in PARAMS["UR"]:
        tards_records[ur] = {
            "RL":[],
            "EDD":[],
            "MS":[],
            "SRO":[],
            "CR":[],
            "noname_2":[],
        }
        ur_records[ur] = {
            "RL":[],
            "EDD":[],
            "MS":[],
            "SRO":[],
            "CR":[],
            "noname_2":[],
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
            tards_record,ur_record,actions = eval_mappo(env_rl,mappo)
            print(f"UR:{ur} {job} total tard:{np.sum(tards_record)},actions:{actions}")
            tards_records[ur]["RL"].append(tards_record)
            ur_records[ur]["RL"].append(ur_record)

    
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
                tards_record,ur_record = eval_sr(env_sr,sr)
                print(f"SR:{sr} UR:{ur} {job} total tard:{np.sum(tards_record)}")
                tards_records[ur][sr].append(tards_record)
                ur_records[ur][sr].append(ur_record)
            
    with open("experiment/jsp/resurt_all.json", "w") as f:
        records = {
            "tards": tards_records,
            "ur": ur_records,
        }
        json.dump(records, f)