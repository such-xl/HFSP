import os
from scheduling_env.training_env import TrainingEnv
from scheduling_env.utils import Plotter
# 创建并初始化环境，从文件中解析job和machine信息，对环境初始化
env = TrainingEnv()
plotter = Plotter(0)
job_file_path = os.path.dirname(os.path.abspath(__file__))+'/scheduling_env/data/Job_Data/Brandimarte_Data/Text/Mk10.fjs'
env.get_jobs_from_file(job_file_path)
flag = True                     #用来判断while循环结束
time_step = 0
while flag:
    #print(f'time_step:{env.time_step}')
    # busy_agen
    # t加工一个time_step
    obs,done = env.run_a_time_step()
    # 后让idle_agent选择一个动作
    idle_agent = env.idle_agents.head
    while idle_agent:                       #遍历所有idle agent
        next_agent = idle_agent.next
        # if env.pending_jobs.length== 6 and env.in_progress_jobs.length==0:
        #     print(' ')
        act_jobs,act_jobs_id = env.get_agent_actions(idle_agent.id)   #获取agent的动作

        action = idle_agent.sample_action(obs,act_jobs,act_jobs_id)          #采样一个动作
        obs,reward,done,info = env.step(idle_agent,action,act_jobs)         #执行动作
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

