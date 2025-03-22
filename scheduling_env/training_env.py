import random
import numpy as np
from .job import Job, JobList, fetch_job_info
from .machine import Machine, MachineList
from .basic_scheduling_algorithms import SPT, LPT, SRPT, LRPT, noname_2

np.random.seed(42)


class TrainingEnv:
    # 初始化环境
    def __init__(
        self,
        obs_dim,
        obs_len,
        state_dim,
        state_len,
        action_dim,
        max_job_num,
        job_file_path,
    ) -> None:
        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.state_dim = state_dim
        self.state_len = state_len
        self.action_dim = action_dim
        self.max_job_num = max_job_num
        self.machine_num, self.job_type = fetch_job_info(path=job_file_path)
        self.job_arrivals = self.create_job_arriavl_seq()

    def create_job_arriavl_seq(self, lambda_rate=0.1):
        """
        生成指数分布的间隔时间，并取整
        """
        intervals = np.random.exponential(
            scale=1 / lambda_rate, size=self.max_job_num - 10
        )
        intervals = np.round(intervals).astype(int)  # 取整转换为整数
        arrival_times = np.cumsum(intervals)
        arrival_times = np.insert(arrival_times, 0, [0] * 10)
        selected_jobs = self.job_type.copy()
        selected_jobs.extend([random.choice(self.job_type) for _ in range(self.max_job_num-10)])
        arrivals = [(job, time) for job, time in zip(selected_jobs, arrival_times)]
        arrivals.sort(key=lambda x: x[1])
        return arrivals

    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_arrivals[self.job_num][1]
        ):
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
        self.idle_action = 0
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.makespans = [0 for _ in range(self.machine_num)]
        self.makespan_max = 0
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.insert_job()
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
        for i in range(30 - self.uncomplete_job.length): # todo list
            global_state.append([0 for _ in range(self.obs_dim)])
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
        truncated = True if self.time_step > 2500 else False
        while (
            not done and not truncated and not self.is_any_machine_need_to_decision()
        ):  # 没有结束且没有空闲机器，继续
            done, truncated = self.run()
        return done, truncated

    def get_obs_i(self):
        """
        获取machine i 的 obs
        如果可用作业大于5，则用调度规则选取的作业信息作为state
        否则
        """
        if len(self.available_jobs) <= self.obs_len - 1:
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

        obs_mask = [False if i < len(obs_i) else True for i in range(self.obs_len)]
        obs_mask[-1] = False
        for i in range(self.obs_len - 1 - len(obs_i)):
            obs_i.append([0 for _ in range(self.obs_dim)])
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
        # makespans_copy = [x for x in self.makespans]

        # expected_job = noname_2(self.available_jobs,self.current_machine,self.compute_UR())
        # expected_reward = self.compute_single_reward(expected_job,[x for x in self.makespans])

        # real_job = self.available_jobs[action] if action<len(self.available_jobs) else None
        # real_reward = self.compute_single_reward(real_job,[x for x in self.makespans])
        # reward = (real_reward - expected_reward) * 30
        reward_idle = 0
        if action == self.action_dim - 1:  # 空闲动作
            # raise("hello")
            # reward = reward * 0.1 if reward >0 else reward*10
            self.idle_action += 1
            reward_idle = -0.005
        else:
            self.current_machine.load_job(self.available_jobs[action], self.time_step)
            self.makespan_max = np.max(self.makespans)
            self.makespans[self.current_machine.id - 1] = (
                self.time_step
                + self.available_jobs[action].current_progress_remaining_time()
            )
        self.current_machine.update_decision_time(self.time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        # reward = self.compute_single_reward(self.current_machine.id)
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
            obs_i, obs_mask = self.get_obs_i()
        else:
            obs_i = [[0 for _ in range(self.obs_dim)] for _ in range(self.obs_len)]
            obs_mask = [True for _ in range(self.obs_len)]
        global_state, state_mask = self.get_global_state()
        # reward = self.compute_single_reward(self.current_machine.id,makespans_copy)
        if truncated:
            reward = -100

        reward = self.compute_single_reward(
            self.current_machine.id,
            self.available_jobs[action] if action < len(self.available_jobs) else None,
        )
        return obs_i, obs_mask, global_state, state_mask, reward, done, truncated

    def step_by_sr(self, action):

        self.current_machine.load_job(action, self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
        return 0, done, truncated

    def is_any_machine_need_to_decision(self):
        for machine in self.machines:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
        return False

    def compute_single_reward1(self, job, makespans_copy, lamda_1=1, lamda_2=1):
        """
        计算单个agent的reward
        """
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        machine = self.current_machine
        index = machine.id - 1
        if job is None:
            u_i = utiliaction_rates[machine.id - 1]
        else:
            u_i = (machine.busy_time + job.get_t_process(machine.id)) / (
                self.time_step + job.get_t_process(machine.id)
            )
        utiliaction_rates[index] = u_i

        u_mean = np.mean(utiliaction_rates) + 1e-6
        t_process = 0 if job is None else job.get_t_process(machine.id)
        s_i = makespans_copy[index] + t_process
        s_max = np.max(makespans_copy)
        reward_1 = s_max - s_i
        reward_2 = np.abs(u_mean - u_i)
        return lamda_1 * reward_1 - lamda_2 * reward_2

    def compute_single_reward(self, machine_id,job=None):
        """
        计算单个agent的reward
        """
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        machine = self.machines[machine_id - 1]
        index = machine.id - 1
        u_i = utiliaction_rates[machine.id - 1]
        utiliaction_rates[index] = u_i

        u_mean = np.mean(utiliaction_rates) + 1e-6
        t_process = 0 if machine.job is None else machine.job.get_t_process(machine.id)

        reward = -np.abs(u_i - u_mean) / u_mean
        if job is None and u_i < u_mean:
            reward -= 0.01
        return reward

    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        return utiliaction_rates
