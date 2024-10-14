import os
import torch
import random
from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import Plotter
from scheduling_env.replay_buffer import ReplayBuffer
from scheduling_env.agents import Agent

"""test"""
#from scheduling_env.model import JobMachineAttentionModel
# 创建并初始化环境，从文件中解析job和machine信息，对环境初始化
env = TrainingEnv()
plotter = Plotter(False)
#self,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon,learning_rate,gamma,target_update, device
lr = 2e-6
num_episodes = 10
job_input_dim  = 128
machine_input_dim = 5
job_hidden_dim = machine_hidden_dim = 128
action_dim = 30
num_heads = 8
job_seq_len = 30
machine_seq_len = 30
gamma = 0.99
epsilon_start = 0
epsilon_end = 0
epsilon_decay = 1000
tau = 0.005
target_update = 100
buffer_size = 100000
minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
job_file_path = os.path.dirname(os.path.abspath(__file__))+'/scheduling_env/train_data/la35.fjs'
replay_buffer  = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
agent.load_model('model.pth')
step_done  = 0
for i in range(num_episodes): #episodes
    #Generate an FJSS instance from teh emulating environment
    s_p_m,s_p_j,s_o_j,idle_agent,act_jobs = env.reset(jobs_path=job_file_path)
    done = False
    while idle_agent and not done:
        # smmple a action
        action = agent.take_action(s_p_m,s_p_j,s_o_j,act_jobs,step_done)
        step_done += 1
        # execute action
        n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,reward,done = env.step(idle_agent,action,act_jobs)
        idle_agent = next_idle_agent
        act_jobs = next_act_jobs
        s_p_m,s_p_j,s_o_j = n_s_p_m,n_s_p_j,n_s_o_j    
    print('time:', env.time_step)