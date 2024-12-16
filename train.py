import os
import time

import numpy as np
import torch
import random
import json

from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import StateNorm, Plotter
from scheduling_env.replay_buffer import ReplayBuffer
from scheduling_env.agents import Agent
from scheduling_env import basic_scheduling_algorithms


class Train():
    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/'

    def train_model(self, model_params: dict, train_params: dict):

        env: TrainingEnv = TrainingEnv(
            action_dim=model_params["action_dim"],
            reward_type=train_params["reward_type"],
            max_machine_num=model_params["machine_seq_len"],
            max_job_num=model_params["job_seq_len"]
        )

        state_norm = StateNorm(
            machine_seq_len=model_params["machine_seq_len"],
            machine_dim=model_params["machine_dim"],
            job_seq_len=model_params["job_seq_len"],
            job_dim=model_params["state_dim"],
            action_dim=model_params["action_dim"],
            scale_factor=train_params["scale_factor"]
        )
        agent: Agent = Agent(model_params, train_params)

        replay_buffer: ReplayBuffer = ReplayBuffer(
            capacity=train_params["buffer_size"],
            state_seq_len=model_params["job_seq_len"],
            state_dim=model_params["state_dim"],
            machine_action_dim=model_params["machine_dim"] + model_params["action_dim"],
            machine_seq_len=model_params["machine_seq_len"],
        )

        train_data_path = self.data_path + 'train_data/'
        jobs_name = sorted(os.listdir(train_data_path))

        record_makespan = {}
        record_reward = {}
        for name in jobs_name:
            record_reward[name] = []
            record_makespan[name] = []
        step_done = 0
        for i in range(train_params['num_episodes']):  # episodes
            start_time = time.time()
            print('episode:', i)
            G = 0
            # Generate an FJSS instance from teh emulating environment
            # job_name = random.choice(jobs_name)
            job_name = random.choice(['vla19.fjs'])
            job_path = train_data_path + job_name
            state, action_mask = env.reset(jobs_path=job_path)
            done, truncated = False, False
            scale_factor = train_params['scale_factor']
            while not done and not truncated:
                # 采样一个动作
                actions = agent.take_action(state, action_mask, step_done)
                # 执行动作
                next_state, next_action_mask, reward, done, truncated = env.step(actions, scale_factor)
                G += reward
                # 存储经验
                actions = [1 if i == actions else 0 for i in range(model_params['action_dim'])]
                actions = [actions, action_mask]
                # if done or truncated or state != next_state or np.random.rand() < 0.01:  # 要做优先经验回放更好
                replay_buffer.add((state, actions, next_state, reward, done, next_action_mask))
                state, action_mask = next_state, next_action_mask
                step_done += 1

                if replay_buffer.size() >= train_params['minimal_size']:
                    transition = replay_buffer.sample(batch_size=train_params['batch_size'])
                    agent.update(transition)

            record_makespan[job_name].append(env.time_step)
            record_reward[job_name].append(G)
            print('=================================')
            print(job_name)
            print('time:', env.time_step, '||G: ', G, '||\ttimestep/s:', env.time_step / (time.time() - start_time),
                  "||\tepsilon:", agent.eps_threshold)
            print('=================================')

        agent.save_model(f"models/model{train_params['reward_type']}rf.pth")
        with open(f"logs/record{train_params['reward_type']}rf.json", 'w') as json_file:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record, json_file)
            print(f"logs saved named record{train_params['reward_type']}.json")
            print(f"model saved named model{train_params['reward_type']}.pth")


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
    "num_episodes": 30000,
    "batch_size": 2,
    "learning_rate": 2e-6,
    "epsilon_start": 1,
    "epsilon_end": 0.005,
    "epsilon_decay": 50 * 3000,
    "gamma": 1,
    "tau": 0.1,
    "target_update": 5000,
    "buffer_size": 50_000,
    "minimal_size": 10_000,
    "scale_factor": 0.01,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "reward_type": 2,
}
trainer = Train()
trainer.train_model(model_params, train_params)
