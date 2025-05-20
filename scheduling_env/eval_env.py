import numpy as np
from .training_env import TrainingEnv
from .job import Job, JobList
import json
from .machine import Machine
from .basic_scheduling_algorithms import EDD, MS, SRO, CR
from .reward_2 import AsyncTardinessReward


class JSPEvalEnv(TrainingEnv):
    # 初始化环境
    def __init__(
        self,
        state_dim,
        action_dim,
        machine_num,
        max_job_num,
        lambda_rate = 0.1,
        job_file_path = "None",
        seed_list = None,
        ur = [70,80,90],
        data_path = "./experiment/jsp/job_data/"
    ):
        super().__init__(
            state_dim, action_dim, machine_num, max_job_num, lambda_rate, job_file_path, seed_list
        )
        self.machine_num = 10
        self.max_job_num = 28
        self.ur = ur
        self.data_path = data_path
        self.eps_num = -2
    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_list[self.job_num].insert_time
        ):
            self.uncomplete_job.append(self.job_list[self.job_num])
            self.job_num += 1

    def reset(self,seed=None):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        self.time_step, self.job_num = 0, 0
        self.eps_num += 1
        self.episode_reward = 0
        UR = self.ur[self.eps_num // 100]
        job_file_path = self.data_path + str(UR)+"/" + "test"+str(self.eps_num%100)
        self.rng = np.random.RandomState(self.seed_list[self.eps_num])
        self.count_actions = [0 for _ in range(self.action_dim)]
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.reward_calculator = AsyncTardinessReward(self.machine_num)
        jobs = None
        self.job_list = []
        with open(job_file_path, "r") as f:
            jobs = json.load(f)
        for i, job in enumerate(jobs, 1):
            process = []
            for machine_id, process_time in zip(job["machine"], job["process"]):
                process.append({machine_id + 1: process_time})
            self.job_list.append(
                Job(
                    i,
                    i,
                    len(job["process"]),
                    process,
                    int(job["insert_time"]),
                    int(job["due_time"]),
                )
            )
        self.job_list.sort(key=lambda x: x.insert_time)
        self.insert_job()
        self.pre_avg_urgency = np.mean(self.compute_urgency())
        self.current_machine = self.get_decision_machines()
        self.available_jobs = self.get_available_jobs()
        obs = self._get_obs()
        info = {}
        return obs,info
