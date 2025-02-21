'''
    多智能体作业调度训练环境
    1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
    2: 判断所有job是否完成 over if done else repeat 1
'''

import math
import random
from .job import Job,JobList
from .machine import Machine,MachineList
from .basic_scheduling_algorithms import SPT,LPT,SRPT,LRPT

class TrainingEnv():
    # 初始化环境
    def __init__(self,action_dim,reward_type,max_machine_num,max_job_num) -> None:
        self._action_space = (0,action_dim-1)
        self._action_dim = action_dim 
        self._machine_num = 0        #总agent数
        self._job_num  = 0         #总作业数
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
        self._time_step = 0
        self._reward_type = reward_type
        self._job_list:list[Job] = None 
        self._machines = None
        self._jobs:JobList = JobList()
        self._current_machine = None
        self.draw_data = None
        self.spans = None
        self.span = 0
    def get_jobs_from_file(self, jobs_path:str):
        self._machine_num = self._jobs.fetch_jobs_from_file(jobs_path)
        self.spans = [0 for _ in range(self._machine_num)]
        self._job_num = self._jobs.length
        self._machines = MachineList(self._machine_num)
        machine:Machine = self._machines.head
        self._machine_list = []
        while machine:
            self._machine_list.append(machine)
            machine = machine.next

    def is_decision_machine(self,machine):
        """
        是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策
        """
        if not machine.is_idle() or machine.step_decision_made(self._time_step):
            return False
        job :Job = self._jobs.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(machine.id):
                return True
            job = job.next
        return False
    def get_decsion_machines(self):
        """
            获取需要做出决策的机器
        """
        decision_machines = []
        machine:Machine = self._machines.head
        while machine:
            if self.is_decision_machine(machine):
                decision_machines.append(machine)
            machine = machine.next
        return decision_machines
    def reset(self,jobs_path:str):
        """
            重置环境
            reutrn:
                state: 当前job环境状态
                machine_action: 决策机器的状态
                action_mask: 机器的动作mask
        """
        self._jobs = JobList()
        self._machines = None
        self.get_jobs_from_file(jobs_path) #从文件中获取job和machine信息
        self._job_list = []
        self._makespan_i = [0 for _ in range(self._machine_num)]
        decision_machines = self.get_decsion_machines()
        self._current_machine = decision_machines[0]
        # job_list
        job:Job = self._jobs.head
        while job:
            self._job_list.append(job)
            job = job.next
        self._time_step = 0
        state  = self.get_state_i(self._current_machine.id)
        self.draw_data = [[] for _ in range(self._job_num)] 
        return state
    def run(self):
        """
            所有忙碌agent和job更新一个time_step,使得必产生空闲机器
            在内添加随机时间
        """
        # 更新one timestep时序
        min_run_timestep = 1
        machine:Machine = self._machines.head
        while machine:
            if machine.is_idle():
                machine = machine.next
                continue
            job:Job = machine.job
            machine.run(min_run_timestep,self._time_step)
            machine = machine.next
        self._time_step += min_run_timestep
        done = True
        job:Job = self._jobs.head
        while job:
            if not job.is_completed():
                done =  False
                break
            job = job.next
        truncated = False if self._time_step < 1500 else True
        while not done and not truncated and not self.is_any_machine_need_to_decision(): # 没有结束且没有空闲机器，继续
            done,truncated = self.run()
        return done,truncated

    def get_state_i(self,macine_id):
        """
            获取macine i 的 obs
        """
        state_i = [
            self._job_list[SPT(self._job_list,macine_id)].get_state_code(),
            self._job_list[LPT(self._job_list,macine_id)].get_state_code(),
            self._job_list[SRPT(self._job_list,macine_id)].get_state_code(),
            self._job_list[LRPT(self._job_list,macine_id)].get_state_code()
        ]
        return state_i
    def step(self,action):
        if action == self._action_dim - 1:
            ...
        else:
            if action == 0:
                job_index = LPT(self._job_list,self._current_machine.id)
            elif action == 1:
                job_index = SPT(self._job_list,self._current_machine.id)
            elif action == 2:
                job_index = LRPT(self._job_list,self._current_machine.id)
            elif action == 3:
                job_index = SRPT(self._job_list,self._current_machine.id)
        
            self._current_machine.load_job(self._job_list[job_index],self._time_step)
        self._current_machine.update_decision_time(self._time_step)
        done,truncated = False,False
        if not self.is_any_machine_need_to_decision(): # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done,truncated = self.run()
        # 要么结束，要么有机器需要采样动作
        if truncated:
            reward = -1000
        elif done:
            reward = 8000/self._time_step
        else:
            reward = -0.01
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self._current_machine = decision_machines[0]
            state_i = self.get_state_i(self._current_machine.id) 
        else:
            state_i = [[0 for _ in range(6)] for _ in range(4)]
        return state_i,reward,done,truncated

    def is_any_machine_need_to_decision(self):
        machine:Machine = self._machines.head
        while machine:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
            machine = machine.next
        return False
    @property
    def action_space(self):
        return self._action_space
    @property
    def jobs_num(self):
        return self._jobs_num
 
    @property
    def time_step(self):
        return self._time_step