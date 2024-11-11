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
                    num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,scale_factor,device):
        env = TrainingEnv(action_dim=action_dim,reward_type=reward_type,max_machine_num=machine_seq_len,max_job_num=job_seq_len)
        state_norm = StateNorm(machine_seq_len,machine_input_dim,job_seq_len,job_input_dim,action_dim,scale_factor)
        state_deque = deque(maxlen=machine_seq_len)
        train_data_path = self.data_path +'train_data/'
        jobs_name = sorted(os.listdir(train_data_path))
        record_makespan = {}
        record_reward = {}
        for name in jobs_name:
            record_reward[name] = []
            record_makespan[name] = []

        replay_buffer = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len,action_dim)
        agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
        step_done  = 0
        for i in range(num_episodes): #episodes
            print('episode:',i)
            trt = 0
            st = 0
            add_time = 0
            start_time = time.time()
            G = 0
            #Generate an FJSS instance from teh emulating environment
            # job_name = random.choice(jobs_name)
            job_name = random.choice(['Mk10.fjs','Mk06.fjs','Mk03.fjs'])
            job_path = train_data_path+job_name
            decision_machines = env.reset(jobs_path=job_path)
            done = False
            while not done:
                # 序贯决策
                for machine in decision_machines:
                    machine_state,job_state,actions,action_mask = env.get_state(machine,decision_machines)
                    # state预处理
                    machine_padded_state,machine_mask = state_norm.machine_padding(machine_state)

                    job_padded_state,job_mask,action_mask = state_norm.job_padding(job_state,action_mask)
                    # 采样一个动作
                    action = agent.take_action(machine_padded_state,machine_mask,job_padded_state,job_mask,actions,action_mask,step_done)
                    # 提交动作
                    env.commit_action(machine,actions,action) 
                    state_deque.append((machine_padded_state,job_padded_state,machine_mask,job_mask,action_mask,action))
                # 执行
                next_decision_machines,reward,done = env.step(decision_machines,scale_factor) # 获取的是平分的缩放奖励
                G += reward*len(decision_machines)
                decision_machines = next_decision_machines
                step_done += 1
                # print(agent.none_action_count,agent.action_count,agent.none_action_count/agent.action_count)

                #将state存入buffer
                sst = time.time()
                while len(state_deque)>1:
                    state = state_deque.popleft() #(machine_padded_state,job_padded_state,machine_mask,job_mask,action_mask,action)
                    next_state = state_deque[0]
                    at = time.time()
                    replay_buffer.add((*(state+next_state[:-1]),reward,True if done and len(state_deque)==1 else False))
                    ate = time.time()-at
                    add_time += ate
                st += time.time()-sst
                if replay_buffer.size()>=minimal_size:
                    transition = replay_buffer.sample(batch_size=batch_size)
                    tt = agent.update(transition)
                    trt+= tt
            record_makespan[job_name].append(env.time_step)
            record_reward[job_name].append(G) 
            end_time = time.time() - start_time
            print('=================================')
            print(job_name)
            print('time:', env.time_step, '||\ttimestep/s:', env.time_step / (time.time() - start_time))
            print('epsodetime:',end_time,'||\t更新模型时间:',trt,'||\t存储时间:',st, '||\t add时间', add_time,'奖励:', G)

            agent.none_action_count = 0
            agent.action_count = 0
            print('=================================')
        agent.save_model(f'models/model{reward_type}.pth')
        with open(f'logs/record{reward_type}.json', 'w') as json_file:
            record = {}
            record['makespan'] = record_makespan
            record['reward'] = record_reward
            json.dump(record,json_file)
            print(f'logs saved named record{reward_type}.json')
            print(f'model saved named model{reward_type}.pth')


    def basic_scheduling(self,ty,reward_type,machine_seq_len,job_seq_len):
        # plot = Plotter(0)
        env = TrainingEnv(action_dim=action_dim,reward_type=reward_type, max_machine_num=machine_seq_len,max_job_num=job_seq_len)
        all_data_path = self.data_path+'all_data/'

        jobs_name = os.listdir(all_data_path)
        jobs_name = ['vla20.fjs']
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
                    _,_,actions,_ = env.get_state(agent, idle_agents)
                    if ty == 0: #短作业优先
                        action = basic_scheduling_algorithms.sjf(actions,agent.id)
                    elif ty == 1: #随机策略
                        action = basic_scheduling_algorithms.random_action(actions) 
                    # 提交动作
                    env.commit_action(agent,actions,action)
                # 执行
                idle_agents,r,done = env.step()    
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


lr = 1e-6
num_episodes = 3000
job_input_dim  = 32
machine_input_dim = 4
job_hidden_dim = 32
machine_hidden_dim = 32
action_dim = 32
num_heads = 4
job_seq_len = 30
machine_seq_len = 15
gamma = 1
epsilon_start = 1
epsilon_end = 0.005
epsilon_decay = 500
tau = 0.005
target_update = 1000
buffer_size = 10_000
scale_factor = 0.01
minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trainer = Train()
reward_type = [0,1,2]

trainer.train_model(reward_type[2],num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,lr,scale_factor,device)
# trainer.basic_scheduling(0,0,machine_seq_len,job_seq_len)
