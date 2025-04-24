import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scheduling_env.DQN_env import TrainingEnv
from scheduling_env.DQN import DQN

def train_async_mappo(env, dqn, num_episodes=1000):
    reward_history = []
    makespan = [] 
    tradness = []  
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/scheduling_env/data/train_data/"
    )
    record = {
            'makespan': {},
            'trad':[],
            'utiliaction_rates':[],
            'reward':[]
    }
    job_name = "vla39.fjs"
    job_path = data_path + job_name
    for episode in range(num_episodes):
        state= env.reset(job_path)
        done, truncated = False, False
        G=0
        i=0
        while not done and not truncated:
            action = dqn.take_action(state)
            next_state,reward,done, truncated = env.step(action)
            dqn.store((state, action, reward, next_state, done))
            loss = dqn.update()
            state = next_state
            G += reward
            i += 1
        # if episode % 10 == 0:
        #     dqn.update_target_network()
        reward_history.append(G)
        
        print(
            f"Episode {episode}/{num_episodes}:, Total reward: {G:.2f}, Epsilon: {dqn.epsilon:.3f} loss:{loss}" 
        )
    
    # reward_u = savgol_filter((np.array(reward_history))[:,0],window_length=100, polyorder=2)
    # reward_wait = savgol_filter((np.array(reward_history))[:,1],window_length=100,polyorder=2)
    
    reward_u = savgol_filter(reward_history,window_length=100,polyorder=2)
    # makespan = savgol_filter(makespan,window_length=50,polyorder=2)
    plt_fig(reward_u,"G")
    # plt_fig(makespan,"makespan")


    # plt_fig(reward_wait,"reward_trad")
    with open("record_task.json", "w") as f:
        json.dump(record, f)

def plt_fig(data,name):
    fig = plt.figure()
    x = np.arange(len(data)) 
    plt.figure()
    plt.plot(x, data, linestyle="-", color="r", label="Data Trend")

    plt.title(name)
    plt.xlabel("episode")
    plt.ylabel("vlaue")
    plt.savefig(f"HFSD//plt//{name}")
 

PARAMS = {
    "num_episodes": 1000,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "state_dim": 6,
    "local_state_dim": 6,
    "local_state_len": 5,
    "gamma": 1,
    "tau": 0.005,
    "global_state_dim": 6,
    "global_state_len": 15,
    "action_dim": 4,
    "max_machine_num": 15,
    "max_job_num": 15,
    "num_heads": 6,
    "lmbda":0.95,
    "eps":0.3,
    "epochs":10,
    "weights":[0.9,0.1],
    "memory_size": 10000,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
}

env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    max_machine_num=PARAMS["max_machine_num"],
    max_job_num=PARAMS["max_job_num"],
)

dqn = DQN(
    state_dim=PARAMS["state_dim"], 
    action_dim = PARAMS["action_dim"], 
    gamma=PARAMS["gamma"], 
    lr=PARAMS["learning_rate"], 
    batch_size=PARAMS["batch_size"], 
    memory_size=PARAMS["memory_size"],

)

train_async_mappo(
    env=env,
    dqn=dqn,
    num_episodes=PARAMS["num_episodes"],
)