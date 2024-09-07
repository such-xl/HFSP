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
        self._idle_agents:MachineList = None
        self._draw_data = None        #画图信息
        self._time_step = 0
        
    def get_jobs_from_file(self, jobs_path:str):
        self._agents_num = self._pending_jobs.decode_job_flie(jobs_path)
        self._jobs_num = self._pending_jobs.length
        self._idle_agents = MachineList(self._agents_num)

        c_n = self._pending_jobs._head
        print('start')
        while c_n:
            c_n.show()
            c_n = c_n.next
        print('end')
        # head = self.pending_jobs.head
        # while head:
        #     head.show()
        #     head = head.next

        self._draw_data = [[] for i in range(self._jobs_num)]

    def get_agent_actions(self,agent_id):
        act_jobs,act_jobs_id = [None],[0]               #第一个动作是idle    
        pending_job = self._pending_jobs.head
        while pending_job:
            if pending_job.match_machine(agent_id):             #该job可被当前agent加工
                act_jobs.append(pending_job)
                act_jobs_id.append(pending_job.id)
            pending_job = pending_job.next
        return act_jobs,act_jobs_id
         
    # 所有忙碌agent和job更新一个time step
    def run_a_time_step(self):
        obs = []
        done = False
        in_progress_job = self._in_progress_jobs._head
        busy_agent = self._busy_agents._head
        # 显然，忙碌agent与处理中的job数量总是一致的，且一一对应，所有可以用一个循环处理
        while in_progress_job and busy_agent:
            in_progress_job.run_a_time_step()
            busy_agent.run_a_time_step()
            state=in_progress_job.get_job_state()
            if in_progress_job.status == 2:     #工序加工完成，转到待加工链表
                next_job = in_progress_job.next
                self._in_progress_jobs.disengage_node(in_progress_job)
                self._pending_jobs.append(in_progress_job)
                self._draw_data[in_progress_job.id-1][-1][-1] = self._time_step
                in_progress_job = next_job
            elif in_progress_job.status == 0:   #所有工序加工完成，转到已完成链表
                next_job = in_progress_job.next
                self._in_progress_jobs.disengage_node(in_progress_job)
                self._completed_jobs.append(in_progress_job)
                self._draw_data[in_progress_job.id-1][-1][-1] = self._time_step
                in_progress_job = next_job
            else:                               #当前时序，未加工完成
                in_progress_job = in_progress_job.next
            if busy_agent.status == 1:          #工序加工结束，转到idle
                next_agent = busy_agent.next
                self._busy_agents.disengage_node(busy_agent)
                self._idle_agents.append(busy_agent)
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
        if self._pending_jobs.length + self._in_progress_jobs.length == 0:    # 所有job完成
            done = True
        return obs,done
                                
        
    # 
    def step(self,idle_machine,action,act_jobs):
        obs = []
        reward = []
        done = False
        info = []
        if action == 0:         #机器选择空闲,对环境不产生影响
            pass
        else:
            # machine load job
            act_jobs[action].load_to_machine(idle_machine.id)
            idle_machine.load_job(act_jobs[action].id,act_jobs[action].get_t_process(idle_machine.id),act_jobs[action].progress)
            # 节点转移
            self._idle_agents.disengage_node(idle_machine)
            self._busy_agents.append(idle_machine)

            self._pending_jobs.disengage_node(act_jobs[action])
            self.in_progress_jobs.append(act_jobs[action])

            # 统计数据绘图
            self._draw_data[act_jobs[action].id-1].append([idle_machine.id,self._time_step,self._time_step])

        if self._pending_jobs.length + self._in_progress_jobs.length == 0:    # 所有job完成
            done = True
        return obs,reward,done,info

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
    def completed_jobs(self):
        return self._completed_jobs
    @property
    def pending_jobs(self):
        return self._pending_jobs
    @property
    def in_progress_jobs(self):
        return self._in_progress_jobs
    @property
    def faulty_agents(self):
        return self._faulty_agents
    @property
    def idle_agents(self):
        return self._idle_agents
    @property
    def busy_agents(self):
        return self._busy_agents
    @property
    def draw_data(self):
        return self._draw_data
    @property
    def time_step(self):
        return self._time_step
    @time_step.setter
    def time_step(self, time_step):
        self._time_step = time_step
'''
env = TrainingEnv()

env.get_jobs_from_file('data/Job_Data/Barnes/Text/mt10c1.fjs')

c_n = env._uncompleted_jobs._head
while c_n:
    c_n.show()
    c_n = c_n.next
print(env._uncompleted_jobs._job_num)
    
'''