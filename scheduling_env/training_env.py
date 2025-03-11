'''
    完成机器选择作业
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
    
'''
import math
from .job import Job,JobList
import numpy as np
from .machine import Machine,MachineList
class TrainingEnv():
    # 初始化环境
    def __init__(self,action_dim,reward_type,max_machine_num,max_job_num) -> None:
        self._action_space = (0,action_dim-1)
        self._action_dim = action_dim 
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
        self._completed_jobs = JobList()
        self._uncompleted_jobs = JobList()
        self._busy_machines = MachineList(0)
        self._faulty_machines = MachineList(0)
        self._idle_machines:MachineList = None
        self._machine_list = []
        self._machines:MachineList = None
        self._draw_data = None        #画图信息
        self._time_step = 0
        self._reward_type = reward_type
        self._decision_machines:list[Machine] = None # 某时刻参与决策的所有机器
        self._job_list:list[Job] = None # 某时刻未完成的作业列表


    def get_jobs_from_file(self, jobs_path:str):
        self._max_machine_num ,job_info,squ = self._uncompleted_jobs.fetch_jobs_from_file(jobs_path)
        
        print('max_machine_num:',self._max_machine_num)
        print('job_info:',job_info)
        print('squ:',squ)
        self._max_job_num = self._uncompleted_jobs.length
        self._machines = MachineList(self._max_machine_num)
        self._idle_machines = self._machines
        machines_list = self._machines.head
        while machines_list:
            self._machine_list.append(machines_list)
            machines_list = machines_list.next

    
    def is_decision_machine(self,agent_id):
        """是否是需要做出决策的agent,当agent只能选择空闲时,则不需要做出决策"""
        uncompleted_job :Job = self._uncompleted_jobs.head
        while uncompleted_job:
            if uncompleted_job.is_wating_for_machine() and uncompleted_job.match_machine(agent_id):
                return True
            uncompleted_job = uncompleted_job.next
        return False
    
    def run(self):
        # 更新one timestep时序
        min_run_timestep = 1
        busy_machine:Machine = self._busy_machines.head
        while busy_machine:
            busy_job:Job = busy_machine.job
            busy_machine.run(min_run_timestep)
            next_busy_machine = busy_machine.next
            if busy_machine.is_idle() : #机器空闲, 代表工序加工完成
                if busy_job.is_completed(): # 所有工序加工完成
                    self._uncompleted_jobs.disengage_node(busy_job)
                    self._completed_jobs.append(busy_job)
                busy_machine.update_begin_idle_time(self._time_step+min_run_timestep) # 更新开始等待时间
                self._busy_machines.disengage_node(busy_machine)
                self._idle_machines.append(busy_machine)

            elif busy_machine.is_fault(): #机器故障,暂时不实现
                self._busy_machines.disengage_node(busy_machine)
                self._faulty_machines.append(busy_machine)
            busy_machine = next_busy_machine
        self._time_step += min_run_timestep
        
        done = False
        if self._uncompleted_jobs.length == 0:    # 所有job完成
            done = True
            return done
        # 获取需要决策的智能体
        decision_machines = []
        idle_machine = self._idle_machines.head
        while idle_machine:
            if self.is_decision_machine(idle_machine.id):
                decision_machines.append(idle_machine) 
            idle_machine = idle_machine.next
        if len(decision_machines)==0: # 如果没有需要决策的智能体，则继续run
            return self.run()
        self._decision_machines = decision_machines
        return done
   
    def reset(self,jobs_path:str):
        """
            重置环境
            reutrn:
                state: 当前job环境状态
                machine_action: 决策机器的状态
                action_mask: 机器的动作mask
        """
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._decision_machines,self._job_list = [],[]
        idle_machine:Machine = self._idle_machines.head
        while idle_machine:
            if self.is_decision_machine(idle_machine.id):
                self._decision_machines.append(idle_machine)
            idle_machine = idle_machine.next
        #重置job_list
        job:Job = self._uncompleted_jobs.head
        while job:
            self._job_list.append(job)
            job = job.next
        self._time_step = 0
        # static_state = self.get_job_static_state()
        state,machine_action,action_mask = self.get_state()
        return state,machine_action,action_mask
    
    def step(self,actions,machine_action,scale_factor):
        for decision_machine,action in zip(self._decision_machines,actions):
            if action == self._action_dim-1:
                continue
            decision_machine.update_end_idle_time(self._time_step) # 更新结束等待时间
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
        reward = 0 if not done else -self.time_step*0.01
        state,machine_action,action_mask = self.get_state()
        return  state,machine_action,action_mask,reward,done
        
        if self._reward_type == 0:
            reward = self.reward_func_0(scale_factor,done)
        elif self._reward_type == 1:
            reward = self.reward_func_1()
        elif self._reward_type == 2:
            reward = self.reward_func_2(scale_factor,actions) 
    # def get_job_static_state(self): #获取所有作业的加工信息
    #     uncompleted_job:Job = self._uncompleted_jobs.head
    #     state = []
    #     while uncompleted_job:
    #         state.append(uncompleted_job.get_state_encoding(self._max_machine_num))
    #         uncompleted_job = uncompleted_job.next
    #     return state
            
    def get_state(self):
        uncompleted_job:Job = self._uncompleted_jobs.head
        state,machine_action = [],[x.get_state_encoding(4) for x in self._decision_machines]
        action_mask = [[] for i in range(len(self._decision_machines))]
        while uncompleted_job:
            state.append(uncompleted_job.get_state_encoding(self._max_machine_num))
            for i,machines in enumerate(self._decision_machines):
                action_mask[i].append(True if uncompleted_job.is_wating_for_machine() and uncompleted_job.match_machine(machines.id) else False)
            uncompleted_job = uncompleted_job.next
        return state,machine_action,action_mask
         
    
    
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
    def reward_func_2(self,scale_factor,actions):
        """
            机器平均空闲率
        """
        idle_machine:Machine = self._idle_machines.head
        busy_machine:Machine = self._busy_machines.head

        idle_rate = 0
        while idle_machine:
            idle_rate += idle_machine.get_idle_time(self.time_step)
            idle_machine = idle_machine.next
        while busy_machine:
            idle_rate += busy_machine.get_idle_time(self.time_step)
            busy_machine = busy_machine.next
        machine_counts = self._idle_machines.length + self._busy_machines.length
        r1 = -idle_rate/machine_counts*scale_factor if machine_counts else 0
        uncompleted_job:Job = self._uncompleted_jobs.head
        latest_finnish = 0
        while uncompleted_job:
            latest_finnish = max(latest_finnish,uncompleted_job.current_progress_remaining_time())
            uncompleted_job = uncompleted_job.next

        r2 = -math.log(latest_finnish*scale_factor+1) if latest_finnish else 0
        r3 = -actions.count(self._action_dim-1)/len(actions)
        return (r1*0.2+r2*0.7+r3*0.1) + 0.5
    @property
    def action_space(self):
        return self._action_space
    @property
    def jobs_num(self):
        return self._jobs_num
    @property
    def draw_data(self):
        return self._draw_data
    @property
    def time_step(self):
        return self._time_step