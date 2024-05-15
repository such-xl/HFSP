import os
from scheduling_env.training_env import TrainingEnv

# 创建并初始化环境，从文件中解析job和machine信息，对环境初始化
env = TrainingEnv()
job_file_path = os.path.dirname(os.path.abspath(__file__))+'/scheduling_env/data/Job_Data/Barnes/Text/mt10c1.fjs'
env.get_jobs_from_file(job_file_path)


flag = True                     #用来判断while循环结束
time_step = 0                   #时序
while flag:
    print(f'time_step:{time_step}')
    # busy_agent加工一个time_step
    obs,done = env.run_a_time_step()
    # 后让idle_agent选择一个动作
    idle_agent = env.idle_agent.head
    while idle_agent:                       #遍历所有idle agent
        next_agent = idle_agent.next
        act_jobs,act_jobs_id = env.get_agent_actions(idle_agent.id)   #获取agent的动作
        action = idle_agent.sample_actions(obs,act_jobs,act_jobs_id)  #采样一个动作
        obs,reward,done,info = env.step(idle_agent.id,action)         #执行动作
        if done:            #所有作业完成
            flag = False
        idle_agent = next_agent




uj = env.uncompleted_jobs

head = uj.head
while head:
    head.show()
    head = head.next
