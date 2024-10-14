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
    #plotter.gant_chat(env.draw_data)
    # for j in env.draw_data:
    #     for i in j:
    #         print(i,end=' ')
    #     print()

# import matplotlib.pyplot as plt
# x = list(range(len(rewards)))
# plt.plot(x, rewards, label='rewards', marker='o')  # 为每个数据点添加‘o’标记
# plt.plot(x, span_times, label='span_times', marker='^')  # 为每个数据点添加‘^’标记

# # 添加标签和标题
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Multiple Y Data on Same X Axis')
# plt.legend()
# plt.savefig('k.png')
# # 展示图形
# plt.show()

'''
while flag:
    # 加工一个time_step
    busy_agent = env.busy_agents.head
    s_o_m,s_o_j,done = env.run_a_time_step() # 返回正在执行的machine和job的state encode
    # 后让idle_agent选择一个动作
    #idle_agent = env.idle_agents.head
    while idle_agent:                       #遍历所有idle agent
        next_agent = idle_agent.next
        #act_jobs,act_jobs_id = env.get_agent_actions(idle_agent.id)   #获取agent的动作
        #随机一个动作
        action = random.randint(-1,len(act_jobs))
        # 执行一个动作,返回下一个state
        n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,reward,done = env.step(idle_agent,action,act_jobs)

        #放入回放池
        replay_buffer.add(s_o_j,s_p_j,s_p_m,action,reward,n_s_o_j,n_s_p_j,n_s_p_m,done)


        #处理信息 s_o_j s_p_j s_p_m
        s_o_j = torch.tensor(s_o_j) 
        s_p_j = torch.tensor(s_p_j)
        s_p_m = torch.tensor(s_p_m)
        if s_o_j.shape[0] == 0: #当前没有运行的job
            s_o_j = torch.zeros(30,128)
        else:
            zero_padding = torch.zeros(30-s_o_j.shape[0],128)
            s_o_j = torch.cat([s_o_j,zero_padding],dim=0)
        
        if s_p_j.shape[0] == 0: #当前machine没有可运行的job
            s_p_j = torch.zeros(29,128)
        else:
            zero_padding = torch.zeros(29-s_p_j.shape[0],128)
            s_p_j = torch.cat([s_p_j,zero_padding],dim=0)
        result = actor(s_p_j.unsqueeze(0).to(torch.float),s_o_j.unsqueeze(0).to(torch.float),s_p_m.unsqueeze(0).to(torch.float))
        print(result)


        if done:            #所有作业完成
            flag = False
            print(env.pending_jobs.length)
            print(env.in_progress_jobs.length)
            print(env.completed_jobs.length)
            print(env.idle_agents.length)
            print(env.busy_agents.length)
            for j in env.draw_data:
                for i in j:
                    print(i,end=' ')
                print()
            print('tiemstep: ',env.time_step)
            print('timestep:',time_step)
            plotter.gant_chat(env.draw_data)
            break
        idle_agent = next_agent
    env.time_step += 1
    time_step += 1

'''