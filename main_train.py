import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO

def train_async_mappo(
    env, ppo, num_episodes=1000, batch_size=64, epochs=10, max_steps=200
):
    reward_history = []
 
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/scheduling_env/data/train_data/"
    )
    record = {
            'makespan': {},
            'trad':[],
            'utiliaction_rates':[],
            'reward':[]
    }
    job_name = "ela21.fjs"
    job_path = data_path + job_name
    for episode in range(num_episodes):
        global_state,mask= env.reset(job_path)
        done, truncated = False, False
        while not done and not truncated: 
            action = ppo.take_action(global_state,mask)
            next_global_state,next_mask,reward, done, truncated = env.step(action)
            ppo.store_transition(global_state,action,reward,next_global_state,done,mask,next_mask)
            global_state = next_global_state
            mask = next_mask
            
        reward_history.append(reward)
        record["makespan"][f"episode_{episode}"] = env.time_step
        record['utiliaction_rates'].append(reward[0])
        record['trad'].append(reward[1])
        record['reward'].append(reward)
        actor_loss,critic_loss = ppo.update()
        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.5f}, Critic Loss {critic_loss:.5f}, make_span {env.time_step}"
        )
    
    reward_u = savgol_filter((np.array(reward_history))[:,0],window_length=100, polyorder=2)
    reward_wait = savgol_filter((np.array(reward_history))[:,1],window_length=100,polyorder=2)

    plt_fig(reward_u,"reward_u")
    plt_fig(reward_wait,"reward_trad")
    with open("record_task.json", "w") as f:
        json.dump(record, f)

def plt_fig(data,name):

    x = np.arange(len(data)) 

    plt.figure(figsize=(8, 4))
    plt.plot(x, data, linestyle="-", color="r", label="Data Trend")

    plt.title(f"{name}")
    plt.xlabel("episode")
    plt.ylabel("vlaue")
    # plt.legend()
    # plt.grid(True)
    plt.savefig(f"HFSD//plt//{name}")
 


PARAMS = {
    "num_episodes": 1000,
    "batch_size": 32,
    "learning_rate": 6e-6,
    "gamma": 1,
    "tau": 0.005,
    "global_state_dim": 8,
    "global_state_len": 15,
    "action_dim": 15,
    "max_machine_num": 10,
    "max_job_num": 15,
    "num_heads": 6,
    "lmbda":0.95,
    "eps":0.3,
    "epochs":10,
    "weights":[0.9,0.1],
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
}

env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    max_machine_num=PARAMS["max_machine_num"],
    max_job_num=PARAMS["max_job_num"],
)

ppo = PPO(
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["max_job_num"],
    lr=PARAMS["learning_rate"],
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
    batch_size=PARAMS["batch_size"],
    epochs=10,
    max_steps=200,
)