"""
多智能体作业调度训练环境
1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
2: 判断所有job是否完成 over if done else repeat 1
"""

import random
import numpy as np
from .job import Job, JobList
from .machine import Machine, MachineList
from .Instance_Generator import Instance_Generator
from .basic_scheduling_algorithms import SPT, SRPT, LPT, LRPT, CR, Random, EDD
job_rng = random.Random(42)
np.random.seed(42)


class TrainingEnv:
    # 初始化环境
    def __init__(self, action_dim, machine_num,E_ave,new_insert,job_info_list=None, job_arrival_time=None) -> None:
        self._action_space = (0, action_dim - 1)
        self._action_dim = action_dim
        # self._machine_num = 0  # 总agent数
        self._machine_num = machine_num
        self._job_num = 0  # 总作业数
        self._max_machine_num = machine_num
        self._max_job_num = (new_insert + 5) #总作业数量，new_insert + inital_num
        self._time_step = 0
        self.job_list: list[Job] = []
        self._machines = None
        self._jobs: JobList = JobList()
        self._current_machine = None
        self.draw_data = None
        self.spans = None
        self.span = 0
        self.E_ave = E_ave
        self.new_insert = new_insert
        if job_info_list is None or job_arrival_time is None:
            # 原来的随机生成
            self.job_info_list, self.job_arrival_time = Instance_Generator(
                self._machine_num, self.E_ave, self.new_insert
            )
        else:
            # 复用外部传入的同一批数据
            self.job_info_list = job_info_list
            self.job_arrival_time = job_arrival_time

    def insert_job(self):
        if self.job_num < len(self.job_info_list):
            for job in self.job_info_list:
                if job["Insert_time"] == self._time_step:
                    self.job_list.append(
                        Job(
                            job["id"],
                            job["process_num"],
                            job["process_list"],
                            insert_time=self._time_step,
                        )
                    )
                    self.job_num += 1

    def is_decision_machine(self, machine):
        if not machine.is_idle() or machine.step_decision_made(self._time_step):
            return False

        for job in self.job_list:
           if job.is_wating_for_machine() and job.match_machine(machine.id):
                return True 
        return False

    def get_decsion_machines(self):
        """
        获取需要做出决策的机器
        """
        decision_machines = []
        machine: Machine = self._machines.head
        while machine:
            if self.is_decision_machine(machine):
                decision_machines.append(machine)
            machine = machine.next
        # np.random.shuffle(decision_machines)
        return decision_machines 

    def reset(self):
        """
        重置环境
        {job.id job.state,job.processing_num,machin.id,remining_optation_num}
        reutrn:
            state: 当前环境中job的状态信息
            machine_action: 决策机器的状态
        """
        self._time_step = 0
        # machine_num, E_ave, New_insert
        self.job_num = 0  # 实时作业数 
        self.job_list = []
        self.insert_job()

        self._machines = MachineList(self._machine_num)
        machine: Machine = self._machines.head
        self._machine_list = []
        while machine:
            self._machine_list.append(machine)
            machine = machine.next

        decision_machines = self.get_decsion_machines()
        self._current_machine = decision_machines[0]
        self._makespan_i = [0 for _ in range(self._machine_num)]
        local_state = self.get_local_state(self._current_machine)
        self.max_span = 0
        self.tradness_ave = 0
        self.tradness_ave_end = 0
        self.U_ave_end = 0

        return local_state

    def get_global_state(self):
        """
        获取全局状态
        """
        global_state = [
            (
                self.job_list[i].get_state_code()
                if i < len(self.job_list)
                else [0 for _ in range(6)]
            )
            for i in range(self._max_job_num)
        ]
        
        return global_state
    
    def run(self):
        # 更新one timestep时序
        min_run_timestep = 1
        machine: Machine = self._machines.head
        while machine:
            if machine.is_idle():
                machine = machine.next
                continue
            job: Job = machine.job
            machine.run(min_run_timestep, self._time_step)
            machine = machine.next
        self._time_step += min_run_timestep
        
        if self.job_num < self._max_job_num:
            self.insert_job()
        done = True

        for job in self.job_list:
            if not job.is_completed():
                done = False
                break
        truncated = False if self._time_step < 25000 else True
        while (
            not done and not truncated and not self.is_any_machine_need_to_decision()
        ):  
            done, truncated = self.run()
        return done, truncated

    def step(self, action):
        if action == 0:
            # job_index = SPT(self.job_list, self._current_machine.id, self._time_step)
            job_index =SPT(self.job_list, self._current_machine.id)
        elif action == 1:
            job_index = LPT(self.job_list, self._current_machine.id)
        elif action == 2:
            job_index = LRPT(self.job_list, self._current_machine.id)
        elif action == 3:
            job_index = SRPT(self.job_list, self._current_machine.id)
        self._current_machine.load_job(self.job_list[job_index], self._time_step)

        self._current_machine.update_decision_time(self._time_step)

        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()

        reward_U = self.compute_machine_utiliaction()
        reward_Tard = self.compute_job_tardiness()

        rewards = [reward_U, reward_Tard]

        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self._current_machine = decision_machines[0]
            local_state= self.get_local_state(self._current_machine)
        else:
            local_state = [[0 for _ in range(6)] for _ in range(5)]

        return local_state, rewards, done, truncated
    
    def is_any_machine_need_to_decision(self):
        machine: Machine = self._machines.head
        while machine:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
            machine = machine.next
        return False
     
    def get_local_state(self, machine):
        """
        获取macine i 的 obs
        """
        state_i = [
            self.job_list[CR(self.job_list, machine.id,self._time_step)].get_state_code(),
            self.job_list[EDD(self.job_list, machine.id)].get_state_code(),
            self.job_list[LRPT(self.job_list, machine.id)].get_state_code(),
            self.job_list[SRPT(self.job_list, machine.id)].get_state_code(),
            [
                machine.id,
                machine.get_utilization_rate(self._time_step),
                np.mean(
                    [
                        machine.get_utilization_rate(self._time_step)
                        for machine in self._machine_list 
                    ]
                ),
                self._time_step / 1000,
                0,
                0,
            ],
        ]
        return state_i
    
    def sr(self,action):
        if action == 0:
            job_index = SPT(self.job_list, self._current_machine.id)
        elif action == 1:
            job_index = LPT(self.job_list, self._current_machine.id)
        elif action == 2:
            job_index = LRPT(self.job_list, self._current_machine.id)
        elif action == 3:
            job_index = SRPT(self.job_list, self._current_machine.id)
        elif action == 4:
            job_index = Random(self.job_list, self._current_machine.id)
        self._current_machine.load_job(self.job_list[job_index], self._time_step)
        self._current_machine.update_decision_time(self._time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        # 要么结束，要么有机器需要采样动作
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self._current_machine = decision_machines[0]

        return done, truncated

    def compute_machine_utiliaction(self):

        utiliaction_rate = [
            machine.get_utilization_rate(self._time_step) for machine in self._machine_list
        ]

        u_mean = np.mean(utiliaction_rate)

        return u_mean
    
    def compute_job_wait_time(self):
        """
        空闲时间占总时间的比例
        """
        job_idle_time_ratio = [
            job.get_job_idle_time_ratio(self._time_step) for job in self.job_list
        ]
        mean_job_idle_time_ratio = np.mean(job_idle_time_ratio)

        return mean_job_idle_time_ratio
        # job_wait_time = [
        #     job.get_wait_time(self._time_step)  for job in self.job_list
            
        # ]
        # return np.mean(job_wait_time)
    
    def compute_job_tardiness(self):

        tardiness = [
            job.get_e_trad_time(self._time_step) for job in self.job_list
        ]
        
        return -sum(tardiness)
    
    def sum_tardiness(self):
        tardiness =[
            job.completed_time - job.due_time for job in self.job_list
        ] 
        return max(tardiness)
    @property
    def action_space(self):
        return self._action_space

    @property
    def jobs_num(self):
        return self._jobs_num

    @property
    def time_step(self):
        return self._time_step
    @property
    def current_machine(self):
        return self._current_machine
    @property
    def machine_num(self):
        return self._machine_num

