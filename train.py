import os
import torch
import random
from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import Plotter
from scheduling_env.replay_buffer import ReplayBuffer
from scheduling_env.agents import Agent
from scheduling_env import basic_scheduling_algorithms
class Train():
    def __init__(self):
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/'

    def train_model(self,num_episodes,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,job_seq_len,machine_seq_len,
                    num_heads,gamma,epsilon_start,epsilon_end,epsilon_decay,tau,target_update,buffer_size,minimal_size,batch_size,device):
        env = TrainingEnv()
        train_data_path = self.data_path +'train_data/'
        jobs_name = os.listdir(train_data_path)
        replay_buffer  = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
        agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
        # train agent
        step_done  = 0
        for i in range(num_episodes): #episodes
            print('episode:',i)
            #Generate an FJSS instance from teh emulating environment
            job_path = train_data_path+random.choice(jobs_name)
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
                # store the info to replay buffer
                replay_buffer.add(s_p_m,s_p_j,s_o_j,action,reward,n_s_p_m,n_s_p_j,n_s_o_j,done)
                idle_agent = next_idle_agent
                act_jobs = next_act_jobs
                s_p_m,s_p_j,s_o_j = n_s_p_m,n_s_p_j,n_s_o_j    
                # train agent
                if replay_buffer.size()>=minimal_size:
                    bspm,bspj,bsoj,baction,breward,bnspm,bnspj,bnsoj,bdone,bmask_spj,bmask_soj,bmask_nspj,bmask_nsoj = replay_buffer.sample(batch_size=batch_size)
                    transition_dict = {
                        'spms':bspm,
                        'spjs':bspj,
                        'sojs':bsoj,
                        'actions':baction,
                        'rewards':breward,
                        'nspms':bnspm,
                        'nspjs':bnspj,
                        'nsojs':bnsoj,
                        'dones':bdone,
                        'mask_spj':bmask_spj,
                        'mask_soj':bmask_soj,
                        'mask_nspj':bmask_nspj,
                        'mask_nsoj':bmask_nsoj
                    }
                    agent.update(transition_dict=transition_dict)
            print('time:', env.time_step)
            print(job_path)
        agent.save_model('modellog(r).pth')
    def basic_scheduling(self,ty):
        env = TrainingEnv()
        all_data_path = self.data_path+'/all_data/'
        jobs_name = os.listdir(all_data_path)

        for job_name in jobs_name:
            job_path = all_data_path + job_name
            _,_,_,idle_agent,act_jobs = env.reset(jobs_path=job_path)
            done = False
            while idle_agent and not done:
                action = None
                if ty == 0: #短作业优先
                    action = basic_scheduling_algorithms.sjf(act_jobs,idle_agent.id)
                _,_,_,idle_agent,act_jobs,_,done = env.step(idle_agent,action,act_jobs)
            print(job_name,': ', env.time_step)
        print(len(jobs_name))


lr = 2e-6
num_episodes = 2000
job_input_dim  = 128
machine_input_dim = 5
job_hidden_dim = machine_hidden_dim = 128
action_dim = 30
num_heads = 8
job_seq_len = 30
machine_seq_len = 30
gamma = 0.99
epsilon_start = 1
epsilon_end = 0.005
epsilon_decay = 1000
tau = 0.005
target_update = 100
buffer_size = 100000
minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainer = Train()
trainer.basic_scheduling(0)
        
"""test"""
"""
#from scheduling_env.model import JobMachineAttentionModel
# 创建并初始化环境，从文件中解析job和machine信息，对环境初始化
env = TrainingEnv()
plotter = Plotter(False)
#self,job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon,learning_rate,gamma,target_update, device
lr = 2e-6
num_episodes = 2000
job_input_dim  = 128
machine_input_dim = 5
job_hidden_dim = machine_hidden_dim = 128
action_dim = 30
num_heads = 8
job_seq_len = 30
machine_seq_len = 30
gamma = 0.99
epsilon_start = 1
epsilon_end = 0.005
epsilon_decay = 1000
tau = 0.005
target_update = 100
buffer_size = 100000
minimal_size = 1000
batch_size = 512
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
job_file_root_path = os.path.dirname(os.path.abspath(__file__))+'/scheduling_env/train_data/'
jobs_name = os.listdir(job_file_root_path)
replay_buffer  = ReplayBuffer(buffer_size,job_input_dim,job_seq_len,machine_input_dim,machine_seq_len)
agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
step_done  = 0
for i in range(num_episodes): #episodes
    print('episode:',i)
    #Generate an FJSS instance from teh emulating environment
    job_path = job_file_root_path+random.choice(jobs_name)
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
        # store the info to replay buffer
        replay_buffer.add(s_p_m,s_p_j,s_o_j,action,reward,n_s_p_m,n_s_p_j,n_s_o_j,done)
        idle_agent = next_idle_agent
        act_jobs = next_act_jobs
        s_p_m,s_p_j,s_o_j = n_s_p_m,n_s_p_j,n_s_o_j    
        # train agent
        if replay_buffer.size()>=minimal_size:
            bspm,bspj,bsoj,baction,breward,bnspm,bnspj,bnsoj,bdone,bmask_spj,bmask_soj,bmask_nspj,bmask_nsoj = replay_buffer.sample(batch_size=batch_size)
            transition_dict = {
                'spms':bspm,
                'spjs':bspj,
                'sojs':bsoj,
                'actions':baction,
                'rewards':breward,
                'nspms':bnspm,
                'nspjs':bnspj,
                'nsojs':bnsoj,
                'dones':bdone,
                'mask_spj':bmask_spj,
                'mask_soj':bmask_soj,
                'mask_nspj':bmask_nspj,
                'mask_nsoj':bmask_nsoj
            }
            agent.update(transition_dict=transition_dict)
    print('time:', env.time_step)
    print(job_path)
agent.save_model('modellog(r).pth') 
"""