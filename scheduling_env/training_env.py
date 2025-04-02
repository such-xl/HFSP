"""
多智能体作业调度训练环境
1: 每个time_step 先忙碌agent加工一个time_step,后让所有空闲agent选择一个动作
2: 判断所有job是否完成 over if done else repeat 1
"""

import random
import numpy as np
from .job import Job, JobList, fetch_job_info
from .machine import Machine, MachineList

job_rng = random.Random(42)
np.random.seed(42)


class TrainingEnv:
    # 初始化环境
    def __init__(self, action_dim, max_machine_num, max_job_num) -> None:
        self._action_space = (0, action_dim - 1)
        self._action_dim = action_dim
        self._machine_num = 0  # 总agent数
        self._job_num = 0  # 总作业数
        self._max_machine_num = max_machine_num
        self._max_job_num = max_job_num
        self._time_step = 0
        self._job_list: list[Job] = []
        self._machines = None
        self._jobs: JobList = JobList()
        self._current_machine = None
        self.draw_data = None
        self.spans = None
        self.span = 0
        self.job_arriavl_time = self.create_job_arriavl_time(self._max_job_num, 0.1)
        self.job_arriavl_sequence = self.create_jobs_arriavl_sequence(self._max_job_num)

    def create_job_arriavl_time(self, num_jobs, lambda_rate):
        """
        生成指数分布的间隔时间，并取整
        """
        intervals = np.random.exponential(scale=1 / lambda_rate, size=num_jobs - 1)
        intervals = np.round(intervals).astype(int)  # 取整转换为整数

        arrviavl_times = np.cumsum(intervals)
        arrviavl_times = np.insert(arrviavl_times, 0, 0)
        return arrviavl_times

    def create_jobs_arriavl_sequence(self, num_jobs):
        """
        生成作业到达顺序
        """
        job_ids = np.arange(1, num_jobs + 1)
        np.random.shuffle(job_ids)
        return job_ids

    def insert_job(self):
        if self._time_step == self.job_arriavl_time[self.job_num]:
            job = next(
                (
                    j
                    for j in self.job_info_list
                    if j["id"] == self.job_arriavl_sequence[self.job_num]
                ),
                None,
            )
            self._job_list.append(
                Job(
                    job["id"],
                    job["process_num"],
                    job["process_list"],
                    insert_time=self._time_step,
                )
            )
            self.job_num += 1

    def get_jobs_from_file(self, jobs_path: str):
        self._machine_num = self._jobs.fetch_jobs_from_file(jobs_path)
        self.spans = [0 for _ in range(self._machine_num)]
        self.max_span = 0
        self._job_num = self._jobs.length

    def is_decision_machine(self, machine):
        """
        是否是需要做出决策的agent,当agent只能选择空闲时,则不需要做出决策
        """
        if not machine.is_idle() or machine.step_decision_made(self._time_step):
            return False

        for job in self._job_list:
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
        np.random.shuffle(decision_machines)
        return decision_machines  # 打乱顺序，模拟异步决策

    def reset(self, jobs_path: str):
        """
        重置环境
        {job.id job.state,job.processing_num,machin.id,remining_optation_num}
        reutrn:
            state: 当前环境中job的状态信息
            machine_action: 决策机器的状态
        """
        self._time_step = 0
        self._machine_num, self.job_info_list = fetch_job_info(jobs_path)
        self.job_num = 0  # 实时作业数
        self._job_list = []
        for job in self.job_info_list:
            self._job_list.append(
                Job(
                    job["id"],
                    job["process_num"],
                    job["process_list"],
                    insert_time=self._time_step,
                )
            )
        self._machines = MachineList(self._machine_num)
        machine: Machine = self._machines.head
        self._machine_list = []
        while machine:
            self._machine_list.append(machine)
            machine = machine.next

        decision_machines = self.get_decsion_machines()
        self._current_machine = decision_machines[0]
        self._makespan_i = [0 for _ in range(self._machine_num)]
        state, mask= self.get_global_state()
        self.draw_data = [[] for _ in range(self._job_num)]
        self.spans = [0 for _ in range(self._machine_num)]
        self.max_span = 0
        self.rewards = []

        return state,mask

    def get_global_state(self):
        """
        获取全局状态
        """

        global_state = [
            (
                self._job_list[i].get_state_code() 

            )
            for i in range(self._max_job_num)

        ]
        #当前等待的加工工序的作业是空闲的并且需要的机器和当前机器匹配
        mask = [    
                not (self._job_list[i].is_wating_for_machine() and self._job_list[i].match_machine(self._current_machine.id))
                if i < len(self._job_list)
                    else True
                for i in range(self._max_job_num)
        ]
        return global_state,mask
    
    def run(self):
        """
        所有忙碌agent和job更新一个time_step,使得必产生空闲机器
        在内添加随机时间
        """
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
        done = True
        for job in self._job_list:
            if not job.is_completed():
                done = False
                break
        truncated = False if self._time_step < 25000 else True
        while (
            not done and not truncated and not self.is_any_machine_need_to_decision()
        ):  # 没有结束且没有空闲机器，继续
            done, truncated = self.run()
        return done, truncated

    def step(self, action):
        
        self._current_machine.load_job(self._job_list[action], self._time_step)
        self._current_machine.update_decision_time(self._time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        # 要么结束，要么有机器需要采样动作
        # reward_U = self.compute_single_reward(self._current_machine.id)
        if done: #没有完成
            reward_U = self.compute_machine_utiliaction() #机器平均利用率
            reward_Trad = self.compute_job_trad()
        else:
            reward_U = 0 #机器平均利用率
            reward_Trad = 0
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self._current_machine = decision_machines[0]
            state, mask = self.get_global_state()
        else:
            state, mask= self.get_global_state()
        return state, mask, [reward_U,reward_Trad], done, truncated
    
    def is_any_machine_need_to_decision(self):
        machine: Machine = self._machines.head
        while machine:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
            machine = machine.next
        return False
    
    def compute_machine_utiliaction(self):
        utiliaction_rate = [
            machine.get_utilization_rate(self._time_step) for machine in self._machine_list
        ]
        u_mean = np.mean(utiliaction_rate) + 1e-6
        return u_mean
    
    def compute_job_trad(self):
        trad_time = [
            job.get_wait_time_rate(self.time_step) for job in self._job_list
        ]
        return np.mean(trad_time)
        # wait_time = [
        #     job.get_wait_time_rate(self._time_step) for job in self._job_list
        # ]
        # mean_wait = np.mean(wait_time)
        # return -mean_wait

    def compute_single_reward(self, agent_id, lamda_1=0, lamda_2=1):
        """
        计算单个agent的reward
        """
        utiliaction_rates = [
            agent.get_utilization_rate(self._time_step) for agent in self._machine_list
        ]
        u_mean = np.mean(utiliaction_rates) + 1e-6
        u_i = utiliaction_rates[agent_id - 1]
        # return u_mean
        return lamda_1 * (u_i / u_mean) - lamda_2 * (np.abs(u_mean - u_i) / u_mean)

    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self._time_step) for agent in self._machine_list
        ]
        return utiliaction_rates

    
    @property
    def action_space(self):
        return self._action_space

    @property
    def jobs_num(self):
        return self._jobs_num

    @property
    def time_step(self):
        return self._time_step
