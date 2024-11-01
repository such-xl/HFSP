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

    def train_model(self,reward_type,num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,
                    num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,device):
        env = TrainingEnv(action_dim=action_dim,reward_type=reward_type)
        state_norm = StateNorm(job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        state_deque = deque(maxlen=machine_seq_len)
        train_data_path = self.data_path +'train_data/'
        jobs_name = sorted(os.listdir(train_data_path))
        record_makespan = {}
        record_reward = {}
        for name in jobs_name:
            record_reward[name] = []
            record_makespan[name] = []

        replay_buffer = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
        step_done  = 0
        for i in range(num_episodes): #episodes
            print('episode:',i)
            gt = trt = 0
            start_time = time.time()
            G = 0
            #Generate an FJSS instance from teh emulating environment
            job_name = random.choice(jobs_name)
            job_name = 'vla20.fjs'
            job_path = train_data_path+job_name
            decision_machine = env.reset(jobs_path=job_path)
            done = False
            while not done:
                # 序贯决策
                for machine in decision_machine:
                    machine_state,job_state,actions = env.get_state(machine)
                    print('1:',type(machine))
                    # state预处理
                    # machine_padded_state,machine_mask = state_norm.machine_padding(machine_state)
                    job_padded_state,job_mask = state_norm.machine_padding(job_state)
                    # 采样一个动作
                    action = agent.take_action(machine_state,job_state)
                    # 提交动作
                    env.commit_action(machine,actions,action) 
                    #state_deque.append((machine_padded_state,job_padded_state,machine_mask,job_mask,action))
                # 执行
                decision_machine,reward,done = env.step() # 获取的是平分的奖励

                #将state存入buffer
                while False and len(state_deque)>1:
                    state = state_deque.popleft() #(machine_padded_state,job_padded_state,machine_mask,job_mask,action)
                    next_state = state_deque[0]
                    replay_buffer.add((*(state+next_state[:-1]),reward,True if done and len(state_deque)==1 else False))
                
                # machine_state,job_state,machine_mask,job_mask,action,reward,done    
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
        with open(f'logs/record{reward_type}iattn.json', 'w') as json_file:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record,json_file)

    def basic_scheduling(self,ty,reward_type,machine_seq_len,job_seq_len):
        # plot = Plotter(0)
        env = TrainingEnv(action_dim=action_dim,reward_type=reward_type, max_machine_num=machine_seq_len,max_job_num=job_seq_len)
        all_data_path = self.data_path+'all_data/'

        jobs_name = os.listdir(all_data_path)
        jobs_name = ['vla01.fjs']
        # record_makespan = {}
        # record_reward = {}
        jobs_name = sorted(jobs_name)
        for job_name in jobs_name:
            job_path = all_data_path + job_name
            idle_agents= env.reset(jobs_path=job_path)
            done = False
            while not done:
                # 序贯决策
                for agent in idle_agents:
                    _,_,actions = env.get_state(agent)
                    if ty == 0: #短作业优先
                        action = basic_scheduling_algorithms.sjf(actions,agent.id)
                    elif ty == 1: #随机策略
                        action = basic_scheduling_algorithms.random_action(actions) 
                    # 提交动作
                    env.commit_action(agent,actions,action)
                # 执行
                idle_agents,done = env.step()    
                # plot.gant_chat(env.draw_data)
            print(job_name,': ', env.time_step,f'rward:{1}')
            # record_makespan[job_name] = env.time_step
            # record_reward[job_name] = cr
        # with open(f'logs/record_random{reward_type}.json','w') as f:
        #     record = {}
        #     record['makespan'] = record_makespan
        #     record['reward'] = record_reward
        #     json.dump(record,f)
        print(len(jobs_name))


lr = 2e-6
num_episodes = 1
job_input_dim  = 42 
machine_input_dim = 5
job_hidden_dim = 32
machine_hidden_dim = 16
action_dim = 30
num_heads = 2
job_seq_len = 30
machine_seq_len = 15
gamma = 1
epsilon_start = 1
epsilon_end = 1
epsilon_decay = 1000
tau = 0.005
target_update = 1000
buffer_size = 10_000

minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = Train()
reward_type = [0,1,2]

# trainer.train_model(reward_type[2],num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,device)
trainer.basic_scheduling(ty=0,reward_type=reward_type[2],machine_seq_len=machine_seq_len,job_seq_len=job_seq_len)