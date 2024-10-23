import os
import time

import torch
import random
import json
from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import Plotter
from scheduling_env.replay_buffer import ReplayBuffer
from scheduling_env.agents import Agent,Agenta
from scheduling_env import basic_scheduling_algorithms
class Train():
    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/'

    def train_model(self,reward_type,num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,
                    num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,device):
        env = TrainingEnv(reward_type)
        train_data_path = self.data_path +'train_data/'
        jobs_name = os.listdir(train_data_path)
        record_makespan = {}
        record_reward = {}
        for name in jobs_name:
            record_reward[name] = []
            record_makespan[name] = []
        
        replay_buffer = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
        job_hidden_dim1 = 64
        job_hidden_dim2 = 32
        cfc_hidden_dim1 = 512
        cfc_hidden_dim2 = 128 
        # agent = Agenta(job_input_dim,job_hidden_dim1,job_hidden_dim2,machine_input_dim,machine_hidden_dim,cfc_hidden_dim1,
        #                 cfc_hidden_dim2,action_dim,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
        # train agent
        step_done  = 0
        for i in range(num_episodes): #episodes
            print('episode:',i)
            gt = trt = 0
            start_time = time.time()
            G = 0
            #Generate an FJSS instance from teh emulating environment
            job_name = random.choice(jobs_name)
            # job_name = 'rla15.fjs'
            job_path = train_data_path+job_name
            s_p_m,s_p_j,s_o_j,idle_agent,act_jobs = env.reset(jobs_path=job_path)
            done = False
                # all of the idle agents sample & execute a action
                # s_p_m,s_p_j,s_o_j,idle_agent,act_jobs,done = env.run_a_time_step()
            while idle_agent and not done:
                # smmple a action
                action = agent.take_action(s_p_m,s_p_j,s_o_j,act_jobs,step_done)
                step_done += 1
                # execute action
                n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,reward,done = env.step(idle_agent,action,act_jobs)
                G += reward
                # store the info to replay buffer
                        
                idle_agent = next_idle_agent
                act_jobs = next_act_jobs
                s_p_m,s_p_j,s_o_j = n_s_p_m,n_s_p_j,n_s_o_j
                # for a in range(0, action_dim):
                #     if a % (len(act_jobs) + 1) == action % (len(act_jobs) + 1):  # a 与 action同余，动作一致
                replay_buffer.add(s_p_m, s_p_j, s_o_j, action, reward, n_s_p_m, n_s_p_j, n_s_o_j, done)
                # train agent

                if replay_buffer.size()>=minimal_size:
                    transition = replay_buffer.sample(batch_size=batch_size)
                    dnt,tt=agent.update(transition)
                    gt += dnt 
                    trt+= tt
            record_makespan[job_name].append(env.time_step)
            record_reward[job_name].append(G) 
            end_time = time.time() - start_time
            print('time:', env.time_step, 'timestep/s:', env.time_step / (time.time() - start_time))
            print('epsodetime:',end_time,'gt:',gt,'tt',trt,G)
            print(job_path)
        agent.save_model(f'model{reward_type}.pth')
        with open(f'logs/record{reward_type}attn.json', 'w') as json_file:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record,json_file)

    def basic_scheduling(self,ty,reward_type=0):
        env = TrainingEnv(reward_type=reward_type)
        all_data_path = self.data_path+'/all_data/'

        jobs_name = os.listdir(all_data_path)
        record_makespan = {}
        record_reward = {}
        for job_name in jobs_name:
            cr = 0
            job_path = all_data_path + job_name
            _,_,_,idle_agent,act_jobs = env.reset(jobs_path=job_path)
            done = False
            while idle_agent and not done:
                action = None
                if ty == 0: #短作业优先
                    action = basic_scheduling_algorithms.sjf(act_jobs,idle_agent.id)
                if ty == 1: #随机策略
                    action = basic_scheduling_algorithms.random_action(act_jobs)
                _,_,_,idle_agent,act_jobs,r,done = env.step(idle_agent,action,act_jobs)
                cr += r
            print(job_name,': ', env.time_step,f'rward:{cr}')
            record_makespan[job_name] = env.time_step
            record_reward[job_name] = cr
        with open(f'logs/record_random{reward_type}.json','w') as f:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record,f)
        print(len(jobs_name))


lr = 2e-6
num_episodes = 3000
job_input_dim  = 72 
machine_input_dim = 5
job_hidden_dim = 64
machine_hidden_dim = 32
action_dim = 30
num_heads = 2
job_seq_len = 30
machine_seq_len = 1
gamma = 1
epsilon_start = 1
epsilon_end = 0.005
epsilon_decay = 1000
tau = 0.005
target_update = 1000
buffer_size = 10_000

minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = Train()
reward_type = [0,1,2]

trainer.train_model(reward_type[2],num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,device)
#trainer.basic_scheduling(1,reward_type=reward_type[2])
