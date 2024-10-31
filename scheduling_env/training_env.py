'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''
import math
from .job_list import JobList
from .machine_list import MachineList
class TrainingEnv():
    # 初始化环境
    count_action = 1000
    def __init__(self,action_dim,reward_type = 0) -> None:
        self._action_space = None   #(1,30)
        self._action_dim = action_dim 
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
        self._reward_type = reward_type
        self._decision_agent = None # 某時刻参与决策的所有机器
    def get_jobs_from_file(self, jobs_path:str):
        self._agents_num = self._pending_jobs.decode_job_flie(jobs_path)
        self._jobs_num = self._pending_jobs.length
        self._idle_agents = MachineList(self._agents_num)
        self._draw_data = [[] for i in range(self._jobs_num)]

    def get_agent_actions(self,machine):
        act_jobs = []                   
        pending_job = self._pending_jobs.head
        while pending_job:
            if pending_job.match_machine(machine.id):             #该job可被当前agent加工
                act_jobs.append(pending_job)
            pending_job = pending_job.next
        return act_jobs
    def is_decision_agent(self,agent_id):
        """是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策"""
        pending_job = self._pending_jobs.head
        while pending_job:
            if pending_job.match_machine(agent_id):
                return True
            pending_job = pending_job.next
        return False
    def commit_action(self,agent,actions,action):
        """提交智能体动作"""
        if action == self._action_dim-1 or action == len(actions): #选择空闲，不对环境产生影响
            return ...
        job = actions[action]
        agent.load_job(job)
        job.load_to_machine(agent)

        self._idle_agents.disengage_node(agent)
        self._busy_agents.append(agent)
        self._pending_jobs.disengage_node(job)
        self._in_progress_jobs.append(job)

    # 所有忙碌agent和job更新若干time_step,使得必产生空闲机器
    def run(self):
        #找出产生空闲机器的最短运行时间
        in_progress_job = self._in_progress_jobs.head
        min_run_timestep = math.inf if in_progress_job else 1
        while in_progress_job:
            min_run_timestep = min(in_progress_job.t_process-in_progress_job.t_processed,min_run_timestep)
            in_progress_job = in_progress_job.next 
        # 更新min_run_timestep时序
        busy_agent = self._busy_agents.head
        while busy_agent:
            busy_agent.run(min_run_timestep)
            next_busy_agent = busy_agent.next
            if busy_agent.status == 1: #工序加工结束
                self._in_progress_jobs.disengage_node(busy_agent.job) # job的工序加工完成，使该job脱离in_progress_job链表
                self._pending_jobs.append(busy_agent.job) if busy_agent.job.status==2 else self._completed_jobs.append(busy_agent.job) # 若jobs未完成，加入等待加工链表，若加工完成，加入完成链表
                busy_agent.unload_job()
                self._busy_agents.disengage_node(busy_agent)
                self._idle_agents.append(busy_agent)
            
            elif busy_agent.status == 0: #机器故障,暂时不实现
                self._busy_agents.disengage_node(busy_agent)
                self._faulty_agents.append(busy_agent)
            busy_agent = next_busy_agent
        self._time_step += min_run_timestep 
        done = False
        if self._pending_jobs.length + self._in_progress_jobs.length == 0:    # 所有job完成
            done = True
            return [],done
        # 获取需要决策的智能体
        idle_agents = []
        idle_agent = self._idle_agents.head
        while idle_agent:
            if self.is_decision_agent(idle_agent.id):
                idle_agents.append(idle_agent)
            idle_agent = idle_agent.next
        if len(idle_agents)==0: # 如果没有需要决策的智能体，则继续run
            return self.run()
        return idle_agents,done
    def reset(self,jobs_path:str):
        self._draw_data = None
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._completed_jobs = JobList()
        #返回 初始化状态，(第一个idle machine(s_p_m),其可选job(s_p_j)正在执行作业job(s_o_j)   
        self._decision_agent = []
        idle_agent = self._idle_agents.head
        while idle_agent:
            if self.is_decision_agent(idle_agent.id):
                self._decision_agent.append(idle_agent)
            idle_agent = idle_agent.next 
        self._time_step = 0
        return self._decision_agent
    def get_state(machine):
        job_state = []
        # todo 
    def step(self):
        # record = []
        # busy_agent = self._busy_agents.head
        # while busy_agent:
        #     record.append((busy_agent.id,busy_agent.job.id,busy_agent.t_process-busy_agent.t_processed))
        #     busy_agent = busy_agent.next
        # print(record)
        next_idle_agents,done = self.run()
        return  next_idle_agents,done
    
    def reward_func_0(self,action,act_jobs,machine_id):
        """
            correlation coefficient: -0.6032
        """
        reward:float = 0
        if len(act_jobs) == 1: # 仅存在一个空闲动作, 
            reward = -math.log(self.time_step+1)
        else:
            if action == len(act_jobs)-1: # 智能体选择空闲动作
                avg_t = 0
                for job in act_jobs[0:-1]:
                    avg_t += job.get_t_process(machine_id)
                avg_t = avg_t/(len(act_jobs)-1)
                reward = -math.log(avg_t+1)
            else:
                reward = -math.log(act_jobs[action].get_t_process(machine_id))
        return reward
    
    def reward_func_1(self,action,act_jobs,machine_id):
        """
             correlation coefficient:  0.70220
        """
        reward:float = 0
        if len(act_jobs) == 1: 
            reward = 0 # ???0.5 or 1???
        else:
            if action == len(act_jobs)-1:
                reward = 0    
            else:
                est = act_jobs[action].earliest_start_time #实际最早开始时间

                eet = act_jobs[action].get_earliest_end_time() # 理论最早结束时间

                tp = act_jobs[action].get_t_process(machine_id) # 实际需要加工的时间

                at = self._time_step + tp #实际加工结束的时间
                reward = tp/(at-est)
        return reward
        
    
    def reward_func_2(self,action,act_jobs,machine_id):
        """
            1: correlation coefficient:  0.29470
            2: correlation coefficient:  -0.87688
        """
        reward:float = 0
        if len(act_jobs) == 1:
            reward = 0
        else:
            if action == len(act_jobs)-1:
                reward  = 0
            else:
                job = act_jobs[action]
                est = job.pest[job.progress-1] # 理论全局最早开始时间
                eet = job.pest[job.progress]   # 理论全局最早结束时间
                tp = job.get_t_process(machine_id) #实际需要加工的时间
                at  = self._time_step + tp         #实际加工完成的时间

                reward = -(at-est)/tp
        #print('reward:',reward)
        return reward
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
    @property
    def decision_agent(self):
        return self._decision_agent