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
    def __init__(self,action_dim,reward_type,max_machine_num,max_job_num) -> None:
        self._action_space = None   #(1,30)
        self._action_dim = action_dim 
        self._agents_num = 0        #总agent数
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
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
        # self._draw_data = [[] for i in range(self._jobs_num)]

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
        job.load_to_machine(agent,self._time_step)
        # self._draw_data[job.id-1].append([agent.id,self._time_step,self.time_step])
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
        # 更新idle_time:
        idle_agent = self._idle_agents.head
        while idle_agent:
            idle_agent.set_idle_time(min_run_timestep)
            idle_agent = idle_agent.next
        # 更新min_run_timestep时序
        busy_agent = self._busy_agents.head
        while busy_agent:
            busy_agent.run(min_run_timestep)
            next_busy_agent = busy_agent.next
            if busy_agent.status == 1: #工序加工结束
                self._in_progress_jobs.disengage_node(busy_agent.job) # job的工序加工完成，使该job脱离in_progress_job链表
                self._pending_jobs.append(busy_agent.job) if busy_agent.job.status==2 else self._completed_jobs.append(busy_agent.job) # 若jobs未完成，加入等待加工链表，若加工完成，加入完成链表
                busy_agent.job.update_pest(self.time_step+min_run_timestep) #更新job 当前工序的理论最早开始时间
                # self._draw_data[busy_agent.job.id-1][-1][-1] = self._time_step+min_run_timestep
                busy_agent.unload_job()
                self._busy_agents.disengage_node(busy_agent)
                self._idle_agents.append(busy_agent)

            
            elif busy_agent.status == 0: #机器故障,暂时不实现
                self._busy_agents.disengage_node(busy_agent)
                self._faulty_agents.append(busy_agent)
            busy_agent = next_busy_agent
        self._time_step += min_run_timestep

        # 更新pending_job的prst，便以计算job的当前工序延时
        pending_job = self._pending_jobs.head
        while pending_job:
            pending_job.update_prst(self.time_step)
            pending_job = pending_job.next
        
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
        self._decision_agent = []
        idle_agent = self._idle_agents.head
        while idle_agent:
            if self.is_decision_agent(idle_agent.id):
                self._decision_agent.append(idle_agent)
            idle_agent = idle_agent.next 
        self._time_step = 0
        return self._decision_agent
    def get_state(self,machine,decision_machines):
        # machine state:
        machine_state = [machine.get_state_encoding(4)] + [ x.get_state_encoding(4)  for x in decision_machines if x is not machine]
        idle_machine = self._idle_agents.head
        while idle_machine:
            if idle_machine not in  decision_machines:
                machine_state.append(idle_machine.get_state_encoding(4)) 
            idle_machine = idle_machine.next
        # job state:
        '''
        # way 1:
        job_state = [x.get_state_encoding(self._max_machine_num) for x in actions]
        in_progress_job,pending_job =  self._in_progress_jobs.head,self._pending_jobs.head
        while pending_job:
            if pending_job not in actions:
                job_state.append(pending_job.get_state_encoding(self._max_machine_num))
            pending_job = pending_job.next
        while in_progress_job:
            job_state.append(in_progress_job.get_state_encoding(self._max_machine_num))
            in_progress_job = in_progress_job.next
        '''
        # way 2:
        job_state,action_mask,actions = [],[],[]
        in_progress_job,pending_job =  self._in_progress_jobs.head,self._pending_jobs.head
        while pending_job:
            job_state.append(pending_job.get_state_encoding(self._max_machine_num))
            action_mask.append(1 if pending_job.match_machine(machine.id) else 0)
            actions.append(pending_job)
            pending_job = pending_job.next

        while in_progress_job:
            job_state.append(in_progress_job.get_state_encoding(self._max_machine_num))
            action_mask.append(0)
            in_progress_job = in_progress_job.next
        return machine_state,job_state,actions,action_mask
         
    def step(self,decision_machines,scale_factor):
        # todo
        next_idle_agents,done = self.run()
        if self._reward_type == 0:
            reward = self.reward_func_0(scale_factor,done)/len(decision_machines)
        elif self._reward_type == 1:
            reward = self.reward_func_1()/len(decision_machines)
        elif self._reward_type == 2:
            reward = self.reward_func_2(scale_factor)/len(decision_machines)
        else:
            reward = 0
        # next_idle_agents,done = self.run()
        return  next_idle_agents,reward,done
    
    def reward_func_0(self,scale_factor,done):
        """
          每步返回-1的奖励
        """ 
        return 1 if done else -0.01
    def reward_func_1(self):
        """
            返回in_progress_jobs的相对延迟的率
        """
        in_progress_job = self._in_progress_jobs.head
        count = 0
        delay_rate = 0
        while in_progress_job:
            if in_progress_job._t_processed == 0:
                delay_rate += in_progress_job.get_delay_ratio()
                count +=1
            in_progress_job = in_progress_job.next
        return -delay_rate/count if count else 0
    def reward_func_2(self,scale_factor):
        """
            机器平均空闲率
        """
        idle_agents = self._idle_agents.head
        count_agent,idle_rate = 0,0
        while idle_agents:
            idle_rate += idle_agents.get_idle_time()
            count_agent += 1
            idle_agents = idle_agents.next
        r1 =  -idle_rate/count_agent*scale_factor if count_agent else 0
        in_progress_job,pending_job =  self._in_progress_jobs.head,self._pending_jobs.head
        count_job,latest_finnish = 0,0
        while pending_job:
            latest_finnish = max(latest_finnish,pending_job.current_progress_need_time())
            count_job += 1
            pending_job = pending_job.next
        while in_progress_job:
            latest_finnish = max(latest_finnish,in_progress_job.current_progress_need_time())
            count_job += 1
            in_progress_job = in_progress_job.next
        r2 = -math.log(latest_finnish*0.01+1) if latest_finnish else 0
        return (r1+r2)/2
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