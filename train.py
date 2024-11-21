import os
import time
import torch
import random
import json
from collections import deque

from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import StateNorm,Plotter
from scheduling_env.replay_buffer import ReplayBuffer
from scheduling_env.agents import Agent
from scheduling_env import basic_scheduling_algorithms

class Train():
    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/'

    def train_model(self,model_params:dict,train_params:dict):

        env = TrainingEnv(
            action_dim = model_params["action_dim"],
            reward_type = train_params["reward_type"],
            max_machine_num = model_params["machine_seq_len"],
            max_job_num = model_params["job_seq_len"]
        )

        state_norm = StateNorm(
            machine_seq_len = model_params["machine_seq_len"],
            machine_dim = model_params["machine_dim"],
            job_seq_len = model_params["job_seq_len"],
            job_dim = model_params["state_dim"],
            action_dim = model_params["action_dim"],
            scale_factor=train_params["scale_factor"]
        )
        agent = Agent(model_params,train_params)


        replay_buffer = ReplayBuffer(
            capacity = train_params["buffer_size"],
            state_seq_len = model_params["job_seq_len"],
            state_dim = model_params["state_dim"],
            machine_action_dim = model_params["machine_dim"] + model_params["action_dim"],
            machine_seq_len = model_params["machine_seq_len"],    
        )

        train_data_path = self.data_path +'train_data/'
        jobs_name = sorted(os.listdir(train_data_path))

        record_makespan = {}
        record_reward = {}
        for name in jobs_name:
            record_reward[name] = []
            record_makespan[name] = []

        for i in range(train_params['num_episodes']): #episodes
            start_time = time.time()
            print('episode:',i)
            G = 0
            #Generate an FJSS instance from teh emulating environment
            job_name = random.choice(jobs_name)
            # job_name = random.choice(['ela01.fjs'])
            job_path = train_data_path+job_name
            state,machine_action,action_mask = env.reset(jobs_path=job_path)
            state,state_mask = state_norm.job_padding(state)
            machine_action,action_mask = state_norm.machine_action_padding(machine_action,action_mask)
            done = False
            scale_factor = train_params['scale_factor']
            step_done = 0
            while not done:
                # 采样一个动作
                actions,machine_action = agent.take_action(state,state_mask,machine_action,action_mask,step_done)
                # 执行动作
                next_state,next_machine_action,next_action_mask,reward,done = env.step(actions,machine_action,scale_factor)

                next_state,next_state_mask = state_norm.job_padding(next_state)
                next_machine_action,action_mask = state_norm.machine_action_padding(next_machine_action,next_action_mask)
                # 存储经验
                replay_buffer.add((state,state_mask,machine_action,reward,done,next_state,next_state_mask))
                state,state_mask,machine_action,action_mask = next_state,next_state_mask,next_machine_action,next_action_mask
                step_done += 1

                if replay_buffer.size() >= train_params['minimal_size']:
                    transition = replay_buffer.sample(batch_size=train_params['batch_size'])
                    agent.update(transition)

            record_makespan[job_name].append(env.time_step)
            record_reward[job_name].append(G) 
            print('=================================')
            print(job_name)
            print('time:', env.time_step, '||\ttimestep/s:', env.time_step / (time.time() - start_time))
            print('=================================')

        
        agent.save_model(f"models/model{train_params['reward_type']}.pth")
        with open(f"logs/record{train_params['reward_type']}.json", 'w') as json_file:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record,json_file)
            print(f"logs saved named record{train_params['reward_type']}.json")
            print(f"model saved named model{train_params['reward_type']}.pth")

model_params = {
    "state_dim": 18,
    "machine_dim": 4,
    "action_dim": 32,
    "num_heads": 1,
    "job_seq_len": 32,
    "machine_seq_len": 16,
    "dropout": 0.1,
}
train_params = {
    "num_episodes": 100,
    "batch_size": 512,
    "learning_rate": 1e-6,
    "epsilon_start": 1,
    "epsilon_end": 0.005,
    "epsilon_decay": 500,
    "gamma": 1,
    "tau": 0.005,
    "target_update": 1000,
    "buffer_size": 10_000,
    "minimal_size": 1000,
    "scale_factor": 0.01,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "reward_type": 0,
}
trainer = Train()
trainer.train_model(model_params,train_params)
