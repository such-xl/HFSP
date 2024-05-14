'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''
from job_list import JobList
from machine_list import MachineList
class TrainingEnv():
    # 初始化环境
    def __init__(self) -> None:
        self._action_space = None   #动作空间(0,1)连续值，应对不同情况下不同agent动作空间不一致
        self._agents_num = 0        #总agent数
        self._jobs_num  = 0         #总作业数
        self._completed_jobs = JobList()
        self._uncompleted_jobs = JobList()
        self._busy_agent = MachineList(0)
        self._fault_agent = MachineList(0)
        self._idle_agent = None
        
    def get_jobs_from_file(self, jobs_path:str):
        self._uncompleted_jobs.decode_job_flie(jobs_path)
        self._jobs_num = self._uncompleted_jobs.job_num
        self._agent_num = self._uncompleted_jobs.machine_num
        self._idle_agent = MachineList(self._agent_num)
    # 
    def step(self):
        pass

env = TrainingEnv()

env.get_jobs_from_file('data/Job_Data/Barnes/Text/mt10c1.fjs')

c_n = env._uncompleted_jobs._head
while c_n:
    c_n.show()
    c_n = c_n.next
print(env._uncompleted_jobs._job_num)
    
