'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''


from .job_list import JobList
from .machine_list import MachineList
class TrainingEnv():
    # 初始化环境
    def __init__(self) -> None:
        self._action_space = None   #动作空间(0,1)连续值，应对不同情况下不同agent动作空间不一致
        self._agents_num = 0        #总agent数
        self._jobs_num  = 0         #总作业数
        self._completed_jobs = JobList()
        self._pending_jobs = JobList()
        self._in_progress_jobs = JobList()
        self._busy_agents = MachineList(0)
        self._faulty_agents = MachineList(0)
        self._idle_agents = None
        
    def get_jobs_from_file(self, jobs_path:str):
        self._pending_jobs.decode_job_flie(jobs_path)
        self._jobs_num = self._pending_jobs.job_num
        self._agent_num = self._pending_jobs.machine_num
        self._idle_agent = MachineList(self._agent_num)
    
    # 所有忙碌agent和job更新一个time step
    def run_a_time_step(self):
        in_progress_job = self._in_progress_jobs._head
        busy_agent = self._idle_agent._head
        # 显然，忙碌agent与处理中的job数量总是一致的，所有可以用一个循环处理
        while in_progress_job and busy_agent:
            in_progress_job.run_a_time_step()
            busy_agent.run_a_time_step()

            if in_progress_job.status == 2:     #工序加工完成，转到待加工链表
                next_job = in_progress_job.next
                self._in_progress_jobs.disengage_node(in_progress_job)
                self._pending_jobs.append(in_progress_job)
                in_progress_job = next_job
            elif in_progress_job.status == 0:   #所有工序加工完成，转到已完成链表
                next_job = in_progress_job.next
                self._in_progress_jobs.disengage_node(in_progress_job)
                self._completed_jobs.append(in_progress_job)
                in_progress_job = next_job
            else:                               #当前时序，未加工完成
                in_progress_job = in_progress_job.next

            
            if busy_agent.status == 2:          #工序加工结束，转到idle
                next_agent = busy_agent.next
                self._busy_agents.disengage_node(busy_agent)
                self._idle_agent.append(busy_agent)
                busy_agent = next_agent
            elif busy_agent.status == 0:        #故障，相应处理，
                next_agent = busy_agent.next
                self._busy_agents.disengage_node(busy_agent)
                '''
                    故障后相应处理，暂时未做
                '''
                self._faulty_agents.append(busy_agent)
                busy_agent = next_agent
            else:                               #当前时序，未加工完成
                busy_agent = busy_agent.next
                                
    # 
    def step(self):
        pass
    @property
    def action_space(self):
        return self._action_space
    @property
    def agents_num(self):
        return self._agents_num
    @property
    def jobs_num(self):
        return self._jobs_num
    @property
    def agent_num(self):
        return self._agent_num
    @property
    def completed_jobs(self):
        return self._completed_jobs
    @property
    def uncompleted_jobs(self):
        return self._uncompleted_jobs
    @property
    def busy_agent(self):
        return self._busy_agent
    @property
    def faulty_agent(self):
        return self._faulty_agent
    @property
    def idle_agent(self):
        return self._idle_agent
    
'''
env = TrainingEnv()

env.get_jobs_from_file('data/Job_Data/Barnes/Text/mt10c1.fjs')

c_n = env._uncompleted_jobs._head
while c_n:
    c_n.show()
    c_n = c_n.next
print(env._uncompleted_jobs._job_num)
    

'''