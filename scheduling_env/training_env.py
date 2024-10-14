'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''
from .job_list import JobList
from .machine_list import MachineList
import math
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
        self._draw_data = None

    def get_jobs_from_file(self, jobs_path:str):
        self._agents_num = self._pending_jobs.decode_job_flie(jobs_path)
        self._jobs_num = self._pending_jobs.length
        self._idle_agents = MachineList(self._agents_num)
        self._draw_data = [[] for i in range(self._jobs_num)]

    def get_agent_actions(self,agent_id):
        act_jobs,act_jobs_id = [],[]                   
        pending_job = self._pending_jobs.head
        while pending_job:
            if pending_job.match_machine(agent_id):             #该job可被当前agent加工
                act_jobs.append(pending_job)
                act_jobs_id.append(pending_job.id)
            pending_job = pending_job.next
        return act_jobs,act_jobs_id
         
    # 所有忙碌agent和job更新一个time step
    def run_a_time_step(self):
        s_o_j = []
        done = False
        in_progress_job = self._in_progress_jobs.head
        busy_agent = self._busy_agents.head
        # 显然，忙碌agent与处理中的job数量总是一致的，且一一对应，所以可以用一个循环处理
        while in_progress_job and busy_agent:
            in_progress_job.run_a_time_step()
            busy_agent.run_a_time_step()
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
            else:                               #当前工序，未加工完成
                s_o_j.append(in_progress_job.get_job_state())
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
        idle_agent = self._idle_agents.head
        s_p_m = []
        s_p_j = []
        act_jobs = None
        if idle_agent:
            s_p_m = [self._idle_agents.head.get_machine_state()]
            act_jobs,_ = self.get_agent_actions(idle_agent.id)
            for aj in act_jobs:
                s_p_j.append(aj.get_job_state())
        else:
            s_p_m = [[0,0,0,0,0]]
        self._time_step += 1
        return s_p_m,s_p_j,s_o_j,idle_agent,act_jobs,done
    def reset(self,jobs_path:str):
        self._draw_data = None
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._completed_jobs = JobList()
        #返回 初始化状态，(第一个idle machine(s_p_m),其可选job(s_p_j)正在执行作业job(s_o_j)                        
        idle_agent = self._idle_agents.head
        s_p_m = [idle_agent.get_machine_state()]
        act_jobs, _ = self.get_agent_actions(idle_agent.id)
        s_p_j = []
        for aj in act_jobs:
            s_p_j.append(aj.get_job_state())
        self._time_step = 0
        return s_p_m,s_p_j,[],idle_agent,act_jobs   
    # 
    def step(self,idle_machine,action,act_jobs):
        reward = -1
        done = False
        act_jobs.append(0) #代表空闲
        action %= len(act_jobs)
        if len(act_jobs) == 1: #仅有一个空闲动作
            reward = -math.log(self._time_step+0.01)
        elif action == len(act_jobs)-1:         #机器选择空闲,对环境不产生影响
            # 奖励设置 
            need_time = 0
            for job in act_jobs[0:-1]:
                need_time += job.get_t_process(idle_machine.id)
            need_time = need_time/(len(act_jobs)-1) + 1 
            reward = -math.log(need_time)
        else:
            # machine load job
            act_jobs[action].load_to_machine(idle_machine.id)
            idle_machine.load_job(act_jobs[action].id,act_jobs[action].get_t_process(idle_machine.id),act_jobs[action].progress)
            # 节点转移
            self._idle_agents.disengage_node(idle_machine)
            self._busy_agents.append(idle_machine)

            self._pending_jobs.disengage_node(act_jobs[action])
            self.in_progress_jobs.append(act_jobs[action])
            reward = -math.log(act_jobs[action].get_t_process(idle_machine.id))
            # 统计数据绘图
            self._draw_data[act_jobs[action].id-1].append([idle_machine.id,self._time_step,self._time_step])
        next_idle_agent = idle_machine.next
        if next_idle_agent:
            n_s_p_j = []
            n_s_p_m  = [next_idle_agent.get_machine_state()]
            next_act_jobs,_ = self.get_agent_actions(next_idle_agent.id)
            for aj in next_act_jobs:
                n_s_p_j.append(aj.get_job_state())
            n_s_o_j = []
            on_job = self._in_progress_jobs.head
            while on_job:
                n_s_o_j.append(on_job.get_job_state())
                on_job = on_job.next
        else: # 后续没有空闲机器，则调用run_a_time_step()直到出现idle
            while True:
                n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,done = self.run_a_time_step()
                if next_idle_agent or done:
                    break
        return n_s_p_m,n_s_p_j,n_s_o_j,next_idle_agent,next_act_jobs,reward,done

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