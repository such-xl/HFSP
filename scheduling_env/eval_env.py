import random
import numpy as np
from .job import Job, JobList
import json
from .machine import Machine
from .basic_scheduling_algorithms import EDD,MS,SRO,CR

np.random.seed(42)


class FJSEvalEnv:
    # 初始化环境
    def __init__(
        self,obs_dim,obs_len,action_dim,max_job_num,file_path) -> None:

        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.machine_num = 10
        self.action_dim = action_dim
        self.max_job_num = max_job_num
        self.file_path = file_path

    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_list[self.job_num].insert_time
        ):
            self.uncomplete_job.append(self.job_list[self.job_num])
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
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        jobs = None
        self.job_list = []
        with open(self.file_path, "r") as f:
            jobs = json.load(f)
        for i,job in enumerate(jobs,1):
            process = []
            for machine_id,process_time in zip(job["machine"],job["process"]):
                process.append({machine_id+1:process_time})
            self.job_list.append(Job(i,i,len(job["process"]),process,int(job["insert_time"]),int(job["due_time"])))
        self.job_list.sort(key=lambda x: x.insert_time)
        self.insert_job()
        decision_machines = self.get_decsion_machines()
        self.current_machine = decision_machines[0]
        self.available_jobs = self.get_available_jobs()
        obs_i, obs_mask = self.get_obs_i()

        return obs_i, obs_mask 

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
            # print(job.is_completed())
            if job.is_completed():
                self.uncomplete_job.disengage_node(job)
                self.complete_job.append(job)
                job.compute_wait_time(self.time_step)
            job = next_job
        # print(f'{self.complete_job.length}:{self.uncomplete_job.length}')
        done = True if self.complete_job.length >= self.max_job_num else False
        truncated = True if self.time_step > 15000 else False
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
        update_avi_jobs = [
            CR(self.available_jobs,self.time_step),
            EDD(self.available_jobs),
            MS(self.available_jobs,self.time_step),
            SRO(self.available_jobs, self.time_step),
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
                np.log(self.time_step) if self.time_step > 1 else 0,
                0,
            ]
        )
        return obs_i, obs_mask

    def step(self, action):

        self.current_machine.load_job(self.available_jobs[action], self.time_step)

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

        reward = 0
        return obs_i, obs_mask, reward, done, truncated

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

    def compute_single_reward(self, machine_id, job=None):
        """
        计算单个agent的reward
        """
        # utiliaction_rates = [
        #     agent.get_utilization_rate(self.time_step) for agent in self.machines
        # ]
        # machine = self.machines[machine_id - 1]
        # index = machine.id - 1
        # u_i = utiliaction_rates[machine.id - 1]
        # utiliaction_rates[index] = u_i

        # u_mean = np.mean(utiliaction_rates) + 1e-6
        # t_process = 0 if machine.job is None else machine.job.get_t_process(machine.id)

        # # reward = -np.abs(u_i - u_mean) / u_mean
        # reward = -(np.max(utiliaction_rates) - np.min(utiliaction_rates))
        return self.reward_function(self.current_machine.id)

    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        return utiliaction_rates

    def compute_idle_time(self):
        idle_times = [agent.get_idle_time(self.time_step) for agent in self.machines]
        return idle_times

    def reward_function(
        self,agent_id
    ):
        # 负载均衡惩罚项
        balance_penalty = self.calc_load_balance_penalty()

        # 利用率熵
        utilization_entropy = self.calc_utilization_entropy()

        # 资源利用动态调整
        # resource_efficiency = self.calc_resource_efficiency()
        ur = self.compute_UR()
        ur_i = ur[agent_id-1]
        ur_mean = np.mean(ur)
        ur_reward =  ur_i - ur_mean if ur_i<ur_mean else 0
        ur_reward = ur_reward / ur_mean if ur_mean > 0 else 0
        # 组合奖励
        reward = 0.1*-balance_penalty + 0.1*utilization_entropy + 0.9*ur_reward
        return reward

    def calc_load_balance_penalty(self):
        utilization_rates = self.compute_UR()

        # 最大最小利用率差异
        load_imbalance = np.max(utilization_rates) - np.min(utilization_rates)

          # 利用率方差
        utilization_variance = np.var(utilization_rates)
        return (load_imbalance + utilization_variance)/ np.mean(utilization_rates) if np.mean(utilization_rates) > 0 else 0

    def calc_utilization_entropy(self):
        utilization_rates = self.compute_UR()
        total_utilization = np.sum(utilization_rates)

        # 避免除零
        probabilities = utilization_rates / (total_utilization + 1e-8)

        # 计算熵
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        normalized_entropy = entropy / np.log(len(utilization_rates))
        return normalized_entropy

    def calc_resource_efficiency(self):
        # 机器空闲时间惩罚
        idle_time_penalty = np.mean(self.compute_idle_time())
        return -(idle_time_penalty)
