'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''
import math
from .job_list import JobList
from .machine_list import MachineList
from .utils import StateNorm
class TrainingEnv():
    # 初始化环境
    def __init__(self,action_dim,reward_type,max_machine_num,max_job_num) -> None:
        self._action_space = (0,action_dim-1)
        self._action_dim = action_dim 
        self._machine_num = 0        #总agent数
        self._job_num  = 0         #总作业数
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
        self._completed_jobs = JobList()
        self._uncompleted_jobs = JobList()
        self._busy_machines = MachineList(0)
        self._faulty_machines = MachineList(0)
        self._idle_machines:MachineList = None
        self._draw_data = None        #画图信息
        self._time_step = 0
        self._reward_type = reward_type
        self._decision_machines = None # 某時刻参与决策的所有机器
        self._job_list = None # 某时刻未完成的作业列表
    def get_jobs_from_file(self, jobs_path:str):
        self._machine_num = self._uncompleted_jobs.fetch_jobs_from_file(jobs_path)
        self._job_num = self._uncompleted_jobs.length
        self._idle_machines = MachineList(self._machine_num)
        # self._draw_data = [[] for i in range(self._jobs_num)]


    def is_decision_machine(self,agent_id):
        """是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策"""
        uncompleted_job = self._uncompleted_jobs.head
        while uncompleted_job:
            if uncompleted_job.is_wating_for_machine() and uncompleted_job.match_machine(agent_id):
                return True
            uncompleted_job = uncompleted_job.next
        return False
    def run(self):
        """
            所有忙碌agent和job更新若干time_step,使得必产生空闲机器
        """
        #找出产生空闲机器的最短运行时间

        min_run_timestep = math.inf
        for job in self._job_list:
            if job.status == 1:
                min_run_timestep = min(min_run_timestep,job.get_process_remaining_time())
        # 更新idle_time:
        idle_machine = self._idle_machines.head
        while idle_machine:
            idle_machine.set_idle_time(min_run_timestep)
            idle_machine = idle_machine.next
    
        # 更新min_run_timestep时序
        busy_machine = self._busy_machines.head
        while busy_machine:
            busy_machine.run(min_run_timestep)
            next_busy_machine = busy_machine.next
            if busy_machine.status == 1: #工序加工结束
                k = busy_machine.job.is_completed()
                if busy_machine.job.is_completed(): # 所有工序加工完成
                    self._uncompleted_jobs.disengage_node(busy_machine.job)
                    self._completed_jobs.append(busy_machine.job)
                    busy_machine.job.update_pest(self.time_step+min_run_timestep) #更新job 当前工序的理论最早开始时间
                    busy_machine.unload_job()
                self._busy_machines.disengage_node(busy_machine)
                self._idle_machines.append(busy_machine)
            
            elif busy_machine.status == 0: #机器故障,暂时不实现
                self._busy_machines.disengage_node(busy_machine)
                self._faulty_machines.append(busy_machine)
            busy_machine = next_busy_machine
        self._time_step += min_run_timestep
        
        done = False
        if self._uncompleted_jobs.length == 0:    # 所有job完成
            done = True
            return done
        # 获取需要决策的智能体
        self._decision_machines = []
        idle_machine = self._idle_machines.head
        while idle_machine:
            if self.is_decision_machine(idle_machine.id):
                self._decision_machines.append(idle_machine) 
            idle_machine = idle_machine.next
        print(len(self._decision_machines))
        if len(self._decision_machines)==0: # 如果没有需要决策的智能体，则继续run
            return self.run()
        return done
    def reset(self,jobs_path:str):
        """
            重置环境
            reutrn:
                state: 当前job环境状态
                machine_action: 决策机器的状态
                action_mask: 机器的动作mask
                action_jobs: 机器的具体可选job
        """
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._decision_machines,self._job_list = [],[]
        idle_machine = self._idle_machines.head
        while idle_machine:
            if self.is_decision_machine(idle_machine.id):
                self._decision_machines.append(idle_machine)
            idle_machine = idle_machine.next
        job = self._uncompleted_jobs.head
        while job:
            self._job_list.append(job)
            job = job.next
        self._time_step = 0
        state,machine_action,action_mask = self.get_state()
        return state,machine_action,action_mask
    def get_state(self):
        uncompleted_job = self._uncompleted_jobs.head
        state,machine_action = [],[x.get_state_encoding(4) for x in self._decision_machines]
        action_mask = [[] for i in range(len(self._decision_machines))]
        while uncompleted_job:
            state.append(uncompleted_job.get_state_encoding(self._max_machine_num))
            for i,machines in enumerate(self._decision_machines):
                action_mask[i].append(True if uncompleted_job.match_machine(machines.id) else False)
            uncompleted_job = uncompleted_job.next
        
        return state,machine_action,action_mask
         
    def step(self,actions,machine_action,scale_factor):
        for decision_machine,action in  zip(self._decision_machines,actions):
            if action == self._action_dim-1:
                continue
            self._idle_machines.disengage_node(decision_machine)
            self._busy_machines.append(decision_machine)
            decision_machine.load_job(self._job_list[action],self._time_step)
        done = self.run()
        # 更新job_list
        job,job_list = self._uncompleted_jobs.head,[]
        while job:
            job_list.append(job)
            job = job.next
        self._job_list = job_list
        print('uncompleted_jobs:',self._uncompleted_jobs.length)
        print('completed_jobs:',self._completed_jobs.length)
        reward = 0
        state,machine_action,action_mask = self.get_state()
        return  state,machine_action,action_mask,reward,done
    
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