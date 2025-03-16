import random
import numpy as np
from .job import Job, JobList, fetch_job_info
from .machine import Machine, MachineList
from .basic_scheduling_algorithms import SPT, LPT, SRPT, LRPT, noname_2

np.random.seed(42)


class TrainingEnv:
    # 初始化环境
    def __init__(self, action_dim, max_job_num, job_file_path) -> None:
        self.action_dim = action_dim
        self.max_job_num = max_job_num
        self.machine_num, self.job_type = fetch_job_info(path=job_file_path)
        self.job_arrivals = self.create_job_arriavl_seq()

        

    def create_job_arriavl_seq(self, lambda_rate=0.1):
        """
        生成指数分布的间隔时间，并取整
        """
        intervals = np.random.exponential(
            scale=1 / lambda_rate, size=self.max_job_num - 1
        )
        intervals = np.round(intervals).astype(int)  # 取整转换为整数
        arrival_times = np.cumsum(intervals)
        arrival_times = np.insert(arrival_times, 0, 0)
        selected_jobs = [random.choice(self.job_type) for _ in range(self.max_job_num)]
        arrivals = [(job, time) for job, time in zip(selected_jobs, arrival_times)]
        arrivals.sort(key=lambda x: x[1])
        return arrivals

    def insert_job(self):
        while self.job_num<self.max_job_num and self.time_step == self.job_arrivals[self.job_num][1]:
            job_info = self.job_arrivals[self.job_num][0]
            self.uncomplete_job.append(
                Job(
                    id=self.job_num + 1,
                    type=job_info["type"],
                    process_num=job_info["process_num"],
                    process_list=job_info["process_list"],
                    insert_time=self.time_step,
                )
            )
            self.job_num += 1

    def is_decision_machine(self, machine):
        """
        是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策
        """
        if not machine.is_idle() or machine.step_decision_made(self.time_step):
            return False
        job: Job = self.uncomplete_job.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(machine.id):
                return True
            job = job.next
        return False

    def get_decsion_machines(self):
        """
        获取需要做出决策的机器
        """
        decision_machines = [
            machine for machine in self.machines if self.is_decision_machine(machine)
        ]

        np.random.shuffle(decision_machines)
        return decision_machines  # 打乱顺序，模拟异步决策

    def get_available_jobs(self):
        """
        获取可选择的作业列表
        """
        available_jobs = []
        job = self.uncomplete_job.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(
                self.current_machine.id
            ):
                available_jobs.append(job)
            job = job.next
        return available_jobs

    def reset(self):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        self.time_step, self.job_num = 0, 0
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.insert_job()  # 初始化时插入一个作业
        decision_machines = self.get_decsion_machines()
        self.current_machine = decision_machines[0]
        self.available_jobs = self.get_available_jobs()
        obs_i, obs_mask = self.get_obs_i()
        global_state, state_mask = self.get_global_state()

        return obs_i, obs_mask, global_state, state_mask

    def get_global_state(self):
        """
        获取全局状态
        """
        global_state, state_mask = [], []
        job = self.uncomplete_job.head
        while job:
            global_state.append(job.get_state_code())
            state_mask.append(False)
            job = job.next
        for i in range(self.max_job_num - self.uncomplete_job.length):
            global_state.append([0 for _ in range(6)])
            state_mask.append(True)
        if state_mask[0] == True:
            state_mask[0] = False
        return global_state, state_mask

    def run(self):
        """
        所有忙碌agent和job更新一个time_step,使得必产生空闲机器
        在内添加随机时间
        """
        # 更新one timestep时序
        min_run_timestep = 1
        for machine in self.machines:
            if not machine.is_running():
                continue
            job = machine.job

            machine.run(min_run_timestep, self.time_step)
            

        self.time_step += min_run_timestep

        if self.job_num < self.max_job_num:
            self.insert_job()
        job: Job = self.uncomplete_job.head
        while job:
            next_job = job.next
            if job.is_completed():
                self.uncomplete_job.disengage_node(job)
                self.complete_job.append(job)
            job = next_job
        # print(f'{self.complete_job.length}:{self.uncomplete_job.length}')
        done = True if self.complete_job.length >= self.max_job_num else False
        truncated = False
        while not done and not truncated and not self.is_any_machine_need_to_decision():  # 没有结束且没有空闲机器，继续
            done, truncated = self.run()
        return done, truncated

    def get_obs_i(self):
        """
        获取machine i 的 obs
        如果可用作业大于5，则用调度规则选取的作业信息作为state
        否则
        """
        if len(self.available_jobs) <= self.action_dim - 1:
            obs_i = [job.get_state_code() for job in self.available_jobs]

        else:
            update_avi_jobs = [
                SPT(self.available_jobs, self.current_machine.id),
                LPT(self.available_jobs, self.current_machine.id),
                SRPT(self.available_jobs, self.current_machine.id),
                LRPT(self.available_jobs, self.current_machine.id),
                noname_2(self.available_jobs, self.current_machine, self.compute_UR()),
            ]
            obs_i = [job.get_state_code() for job in update_avi_jobs]
            self.available_jobs = update_avi_jobs


        obs_mask = [False if i < len(obs_i) else True for i in range(self.action_dim)]
        obs_mask[-1] = False
        for i in range(self.action_dim-1-len(obs_i)):
            obs_i.append([0 for _ in range(len(obs_i[0]))])
        obs_i.append(
            [
                self.current_machine.id,
                self.current_machine.get_utilization_rate(self.time_step),
                np.mean(self.compute_UR()),
                np.std(self.compute_UR()),
                self.time_step / 100,
                0,
            ]
        )
        return obs_i, obs_mask

    def step(self, action):
        if action == self.action_dim - 1:
            ...
        else:
            self.current_machine.load_job(self.available_jobs[action], self.time_step)

        self.current_machine.update_decision_time(self.time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        reward = self.compute_single_reward(self.current_machine.id)
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
            obs_i, obs_mask = self.get_obs_i()
        else:
            obs_i = [[0 for _ in range(6)] for _ in range(self.action_dim)]
            obs_mask = [True for _ in range(len(obs_i))]
        global_state, state_mask = self.get_global_state()
        return obs_i, obs_mask, global_state, state_mask, reward, done, truncated

    def step_by_sr(self,action):

        self.current_machine.load_job(action, self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        reward = self.compute_single_reward(self.current_machine.id)
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
        return reward, done, truncated

    def is_any_machine_need_to_decision(self):
        for machine in self.machines:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
        return False

    def compute_single_reward(self, agent_id, lamda_1=0, lamda_2=1):
        """
        计算单个agent的reward
        """
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        u_max = np.max(utiliaction_rates) + 1e-6
        u_mean = np.mean(utiliaction_rates) + 1e-6
        u_i = utiliaction_rates[agent_id - 1]

        return lamda_1 * (u_i / u_mean) - lamda_2 * (np.abs(u_mean - u_i) / u_mean)

    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        return utiliaction_rates
