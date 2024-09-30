import os
from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import Plotter
from scheduling_env.model import Actor
from scheduling_env.reply_buffer import ReplayBuffer
from scheduling_env.agents import Agent

#from scheduling_env.model import JobMachineAttentionModel
# 创建并初始化环境，从文件中解析job和machine信息，对环境初始化
ccc = 0
env = TrainingEnv()
plotter = Plotter(0)
job_file_path = os.path.dirname(os.path.abspath(__file__))+'/scheduling_env/data/Job_Data/Dauzere_Data/Text/18a.fjs'
#env.get_jobs_from_file(job_file_path)
actor = Actor(128,128,5,128,1)
replay_buffer  = ReplayBuffer(10000)
agent = Agent(0.01)
flag = True                     #用来判断while循环结束
time_step = 0


for i in range(1): #episodes
    #Generate an FJSS instance from teh emulating environment
    s_p_m,s_p_j,idle_agent,act_jobs = env.reset(jobs_path=job_file_path)
    done = False
    while not done:
        # All of the busy machines run a time step
        s_p_m,s_p_j,s_o_j,idle_agent,act_jobs,done = env.run_a_time_step()
        # all of the idle agents sample & execute a action
        while idle_agent:
            # smmple a action
            action = agent.take_action(s_p_m,s_p_j,s_o_j,act_jobs)
            # execute action
            n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,reward,done = env.step(idle_agent,action,act_jobs)
            # store the info to replay buffer
            replay_buffer.add(s_p_m,s_p_j,s_o_j,action,reward,n_s_p_m,n_s_p_j,n_s_o_j,done)
            idle_agent = next_idle_agent
            act_jobs = next_act_jobs
            s_p_m,s_p_j,s_o_j = n_s_p_m,n_s_p_j,n_s_o_j    
            # train agent
            # if replay_buffer.size()>=100:
            #     pass
        env.time_step+=1
        print('time', env.time_step)
    plotter.gant_chat(env.draw_data)

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