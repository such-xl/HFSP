import os
import time
import random

import torch

from scheduling_env.agents import Agent
from scheduling_env.training_env import TrainingEnv
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

model_params = {
    "state_dim": 12,
    "machine_dim": 10,
    "state_embedding_dim": 12,
    "macihne_embedding_dim": 12,
    "machine_state_dim": 4,
    "action_dim": 11,
    "num_heads": 4,
    "job_seq_len": 10,
    "machine_seq_len": 10,

    "dropout": 0.05,
}
train_params = {
    "num_episodes": 100,
    "batch_size": 256,
    "learning_rate": 5e-3,
    "epsilon_start": 0.00001,
    "epsilon_end": 0.0000001,
    "epsilon_decay": 20_000,
    "gamma": 1,
    "tau": 0.5,
    "target_update": 5000,
    "buffer_size": 50_000,
    "minimal_size": 10_000,
    "scale_factor": 0.01,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "reward_type": 2,
}

agent = Agent(model_params, train_params)
agent.load_model(f"models/model{train_params['reward_type']}rf.pth")
env = TrainingEnv(
            action_dim=model_params["action_dim"],
            reward_type=train_params["reward_type"],
            max_machine_num=model_params["machine_seq_len"],
            max_job_num=model_params["job_seq_len"]
        )
train_data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/' + 'train_data/'

record_reward = []
for i in range(train_params['num_episodes']):  # episodes
    start_time = time.time()
    print('episode:', i)
    G = 0
    job_name = random.choice(['vla19.fjs'])
    job_path = train_data_path + job_name
    state, action_mask = env.reset(jobs_path=job_path)
    done, truncated = False, False
    scale_factor = train_params['scale_factor']
    step = 0
    while not done and not truncated:
        # 采样一个动作
        actions = agent.take_action(state, action_mask, 0)
        # 执行动作
        next_state, next_action_mask, reward, done, truncated = env.step(actions, scale_factor)
        G += reward
        # 存储经验
        actions = [1 if i == actions else 0 for i in range(model_params['action_dim'])]
        actions = [actions, action_mask]
        state, action_mask = next_state, next_action_mask
        step += 1
    print(env._time_step)
    step = env._time_step
    record_reward.append((job_name, i, done, G, step))

import seaborn as sns
import pandas as pd
record_reward_table = pd.DataFrame(record_reward, columns=['dataset', 'episode', 'done', 'Return', 'timestep'])
sns.boxenplot(record_reward_table, x='done', y='timestep', hue='dataset')
plt.savefig('ddlx.png')
print(record_reward_table.groupby(['dataset', 'done']).agg('mean'))