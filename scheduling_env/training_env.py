import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .job import Job, JobList
import json
from .machine import Machine
from .basic_scheduling_algorithms import EDD, MS, SRO, CR, SRPT, FIFO, ATC,PDR_RULES
from .reward_2 import AsyncTardinessReward


class TrainingEnv(gym.Env):
    # 初始化环境
    def __init__(
        self,
        state_dim,
        action_dim,
        machine_num,
        max_job_num,
        lambda_rate,
        job_file_path,
        seed_list,
    ) -> None:
        super(TrainingEnv, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.machine_num = machine_num
        self.max_job_num = max_job_num
        self.seed_list = seed_list
        self.eps_num = -2
        self.rng = np.random.RandomState(self.seed_list[self.eps_num])
        self.lambda_rate = lambda_rate
        self.reward_calculator = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )
        if "fjsp" in job_file_path:
            with open(job_file_path, "r") as f:
                self.job_type = json.load(f)
                # 转换键
            for job in self.job_type:
                for process in job["process_list"]:
                    # 创建新的字典来存储转换后的键
                    new_process = {}
                    for key, value in process.items():
                        new_process[int(key)] = value  # 将字符串键转换为整数
                    # 用转换后的字典替换原字典
                    job["process_list"][
                        job["process_list"].index(process)
                    ] = new_process
            
        # self.job_arrivals = self.create_job_arriavl_seq(self.lambda_rate)

    def create_job_arriavl_seq(self, lambda_rate=0.12):
        """
        生成指数分布的间隔时间，并取整
        """
        intervals = self.rng.exponential(
            scale=1 / lambda_rate, size=self.max_job_num - 10
        )
        intervals = np.round(intervals).astype(int)  # 取整转换为整数
        arrival_times = np.cumsum(intervals)
        arrival_times = np.insert(arrival_times, 0, [0] * 10)
        selected_jobs = [
            self.rng.choice(self.job_type) for _ in range(self.max_job_num)
        ]
        arrivals = [
            (
                job,
                time,
                int(
                    sum(sum(d.values()) / len(d) for d in job["process_list"])
                    * (1.5-self.lambda_rate)
                    + time
                ),
            )
            for job, time in zip(selected_jobs, arrival_times)
        ]
        arrivals.sort(key=lambda x: x[1])
        return arrivals

    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_arrivals[self.job_num][1]
        ):
            job_info = self.job_arrivals[self.job_num][0]
            insert_job = Job(
                id=self.job_num + 1,
                type=job_info["type"],
                process_num=job_info["process_num"],
                process_list=job_info["process_list"],
                insert_time=self.time_step,
                due_time=self.job_arrivals[self.job_num][2],
            )
            self.reward_calculator.update_job_info(
                insert_job.id - 1,
                insert_job.due_time,
                insert_job.get_remaining_avg_time(),
                self.time_step,
            )
            self.uncomplete_job.append(insert_job)
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

    def get_decision_machines(self):
        """
        获取需要做出决策的机器
        """
        decision_machines = [
            machine for machine in self.machines if self.is_decision_machine(machine)
        ]
        index = self.rng.randint(0, len(decision_machines))
        return decision_machines[index]
        # return decision_machines[0]

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

    def reset(self, seed=None, options=None):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        super().reset(seed=seed)
        self.time_step, self.job_num = 0, 0
        self.episode_reward = 0
        self.eps_num += 1
        self.count_actions = [0 for _ in range(self.action_dim)]
        self.rng = np.random.RandomState(self.seed_list[self.eps_num])
        self.job_arrivals = self.create_job_arriavl_seq(self.lambda_rate)
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.reward_calculator = AsyncTardinessReward(self.machine_num)
        self.insert_job()
        self.pre_avg_urgency = np.mean(self.compute_urgency())
        self.current_machine = self.get_decision_machines()
        self.available_jobs = self.get_available_jobs()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        """
        获取当前环境状态
        """
        ranked_job0 = EDD(self.available_jobs,self.time_step,self.current_machine.id)[0]
        ranked_job1 = CR(self.available_jobs, self.time_step,self.current_machine.id)[0]
        ranked_job2 = SRO(self.available_jobs, self.time_step,self.current_machine.id)[0]
        ranked_job3 = ATC(self.available_jobs, self.time_step,self.current_machine.id)[0]

        ranked_job4 = MS(self.available_jobs, self.time_step,self.current_machine.id)[0]
        update_avi_jobs = [ranked_job0, ranked_job1, ranked_job2, ranked_job3,ranked_job4]
        unique_count = len(set(id(obj) for obj in update_avi_jobs))
        self.count_actions[unique_count - 1] += 1
        obs_i = []
        for (
            i,
            job,
        ) in enumerate(update_avi_jobs):
            code1 = job.get_state_code(self.time_step)
            code2 = [1 if j == i else 0 for j in range(4)]
            code = code1 + code2
            obs_i.extend(code)
        self.available_jobs = update_avi_jobs
        all_urgency = self.compute_urgency()
        obs_il = [
            ranked_job0.get_urgency(self.time_step),
            ranked_job1.get_urgency(self.time_step),
            ranked_job2.get_urgency(self.time_step),
            ranked_job3.get_urgency(self.time_step),
            ranked_job4.get_urgency(self.time_step),
            np.mean(all_urgency),
            np.std(all_urgency),
            np.max(all_urgency),
            np.min(all_urgency),
            sum([1 for urgency in all_urgency if urgency == 1]) / len(all_urgency),
        ]
        obs_i.extend(obs_il)
        return np.array(obs_i, dtype=np.float32)

    def run(self):
        """
        所有忙碌agent和job更新一个time_step,使得必产生空闲机器
        在内添加随机时间
        """
        # 更新one timestep时序
        min_run_timestep = 1  # 方便后续引入机器故障等随机事件
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
            machine = job.machine
            next_job = job.next
            self.reward_calculator.update_job_info(
                job.id - 1, job.due_time, job.get_remaining_avg_time(), self.time_step
            )
            if job.is_completed():
                self.uncomplete_job.disengage_node(job)
                self.complete_job.append(job)
                job.compute_wait_time(self.time_step)
                self.reward_calculator.update_job_completion(
                    job.id - 1, self.time_step, machine.id - 1
                )
            job = next_job
        done = True if self.complete_job.length >= self.max_job_num else False
        truncated = True if self.time_step > 15000 else False
        while (
            not done and not truncated and not self.is_any_machine_need_to_decision()
        ):  # 没有结束且没有空闲机器，继续
            done, truncated = self.run()
        return done, truncated

    def step(self, action):
        self.current_machine.load_job(self.available_jobs[action], self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        done, truncated, info = False, False, {}
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        if not done and not truncated:
            self.current_machine = self.get_decision_machines()
            self.available_jobs = self.get_available_jobs()
            obs_i = self._get_obs()
        else:
            obs_i = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "lambad_rate":self.lambda_rate,
                "actions_count": self.count_actions,
                "tardiness": self.get_tardiness()[1],
                "episode_reward": self.episode_reward,
                "seed":self.rng.randint(0,9)
            }
            self.episode_reward = 0
        reward, _ = self.reward_calculator.calculate_system_reward(self.time_step)
        self.episode_reward += reward
        return obs_i, reward, done, truncated, info

    def step_by_sr(self, action):

        self.current_machine.load_job(PDR_RULES[action](self.available_jobs,self.time_step,self.current_machine.id)[0], self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        done, truncated,info = False, False,{}
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        if not done and not truncated:
            self.current_machine = self.get_decision_machines()
            self.available_jobs = self.get_available_jobs()
        else:
            info = {
                "lambda_rate":self.lambda_rate,
                "actions_count": self.count_actions,
                "tardiness": self.get_tardiness()[1],
                "episode_reward": self.episode_reward,
                "seed":self.rng.randint(0,9)
            }
            self.episode_reward = 0 
    
        reward, _ = self.reward_calculator.calculate_system_reward(self.time_step)
        obs_i = [0 for i in range(self.state_dim)]
        return obs_i,reward,done, truncated,info

    def is_any_machine_need_to_decision(self):
        for machine in self.machines:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
        return False

    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        return utiliaction_rates

    def compute_idle_time(self):
        idle_times = [agent.get_idle_time(self.time_step) for agent in self.machines]
        return idle_times

    def compute_slack_time(self):
        slack_time = []
        job = self.uncomplete_job.head
        while job:
            slack_time.append(job.get_slack_time(self.time_step))
            job = job.next
        return slack_time

    def compute_urgency(self):
        urgency = []
        job = self.uncomplete_job.head
        while job:
            urgency.append(job.get_urgency(self.time_step))
            job = job.next
        return urgency

    def get_tardiness(self):
        """
        return
            job_id_list,tardiness_list
        """
        jid, tardiness = [], []
        job = self.complete_job.head
        while job:
            jid.append(job.id)
            tardiness.append(job.tard_time)
            job = job.next
        return jid, tardiness

