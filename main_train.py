import os
import torch
import json
import numpy as np
from params import PARAMS
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO

def train_async_mappo(env, ppo, num_episodes=1000): 
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/scheduling_env/data/train_data/"
    )
    record = {
            'makespan': [],
            'wait_time':[],
            'utiliaction_rates':[],
            'idle_time_ratio': [],
            'reward':[],
            'tradiness': [],
            'flow_time': [],
    }
    for episode in range(num_episodes):
        actions = [0,0,0,0]
        locals_state= env.reset()
        global_state = env.get_global_state()
        done, truncated = False, False
        G=0
        while not done and not truncated:
            action = ppo.take_action(locals_state)
            actions[action] += 1
            next_locals_state,reward,done, truncated = env.step(action)
            next_global_state= env.get_global_state()
            ppo.store_transition(
                locals_state,global_state, action,reward,next_locals_state,next_global_state,done
                )
            global_state = next_global_state
            locals_state = next_locals_state
        tradiness = [
            max((job.completed_time - job.due_time),0) for job in env.job_list
        ]
        
        wait_time = [
            (job.get_wait_time(env.time_step)) for job in env.job_list
        ]

        flow_time = [
            (job.completed_time - job.insert_time) for job in env.job_list
        ]

        record["makespan"].append(env.time_step)
        record['utiliaction_rates'].append(reward[0])
        record['wait_time'].append(np.mean(wait_time))
        record['idle_time_ratio'].append(reward[1])
        record['reward'].append(reward)
        record['tradiness'].append(max(tradiness))
        record['flow_time'].append(np.mean(flow_time))
        actor_loss,loss_U, loss_trad = ppo.update()
        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.4f}, loss_U {loss_U:.4f}, loss_wait {loss_trad:.4f},tradness {sum(tradiness):.4f} wait_time {np.mean(wait_time):.4f} make_span {env.time_step},actions:{actions}"
        )
        
    with open(f"HFSD/record/record_{env.machine_num}_{env.E_ave}_{env.new_insert}_RL.json", "w") as f: # machine E_ave new_insert
        json.dump(record, f)

    ppo.save_model(f"HFSD/models/ppo_model_{env.machine_num}_{env.E_ave}_{env.new_insert}_RL.pth")
    
def step_by_sr(env, num_episodes, action, name):
    record = {
            'makespan': [],
            'wait_time':[],
            'utiliaction_rates':[],
            'reward':[],
            'tradiness': [],
            'idle_time_ratio': [],
            'flow_time': [],
    }
    for episode in range(num_episodes):
        actions = [0,0,0,0]
        _ = env.reset()
        done, truncated = False, False
        G=0
        while not done and not truncated:
            actions[action] += 1
            _,reward,done, truncated = env.step(action)
        
        tradiness = [
            (max(job.completed_time - job.due_time,0)) for job in env.job_list
        ]
        
        wait_time = [
            (job.get_wait_time(env.time_step)) for job in env.job_list
        ]

        flow_time = [
            (job.completed_time - job.insert_time) for job in env.job_list
        ]

        record["makespan"].append(env.time_step)
        record['utiliaction_rates'].append(reward[0])
        record['wait_time'].append(np.mean(wait_time))
        record['idle_time_ratio'].append(reward[1])
        record['reward'].append(reward)
        record['tradiness'].append(sum(tradiness))
        record['flow_time'].append(np.mean(flow_time))
        print(
            f"Episode {episode + 1}/{num_episodes}: make_span {env.time_step}, tradiness {sum(tradiness)} actions {actions}"
        )
            
    
    with open(f"HFSD/record/record_{name}.json", "w") as f:
        json.dump(record, f)

env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    machine_num=PARAMS["machine_num"],
    E_ave=PARAMS["E_ave"],
    new_insert=PARAMS['new_insert']
)

ppo = PPO(
    local_state_dim = PARAMS["local_state_dim"],
    local_state_len = PARAMS["local_state_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    a_lr=PARAMS["actor_lr"],
    c_lr=PARAMS["critic_lr"],
    gamma=PARAMS["gamma"],
    lmbda=PARAMS["lmbda"],
    epochs =PARAMS["epochs"],
    eps=PARAMS["eps"],
    device=PARAMS["device"],
    weights=PARAMS["weights"],
    batch_size = PARAMS["batch_size"]
)

train_async_mappo(
    env=env,
    ppo=ppo,
    num_episodes=PARAMS["num_episodes"],
)


# sr = [ "SPT", "LPT", "LRPT","SRPT"]
# action = 3
# step_by_sr(
#         env=env,
#         num_episodes=PARAMS["num_episodes"],
#         action=action,
#         name = sr[action]
#     )
