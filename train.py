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
    obs,done = env.working_one_time_step()


    # 后让idle_agent选择一个动作






uj = env.uncompleted_jobs

head = uj.head
while head:
    head.show()
    head = head.next
