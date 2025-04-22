import numpy as np
from .training_env import TrainingEnv
from .job import Job, JobList
import json
from .machine import Machine
from .basic_scheduling_algorithms import EDD, MS, SRO, CR
from .reward_2 import AsyncTardinessReward


class FJSEvalEnv(TrainingEnv):
    # 初始化环境
    def __init__(
        self,
        obs_dim,
        obs_len,
        state_dim,
        state_len,
        action_dim,
        machine_num,
        max_job_num,
        file_path,
        rng,
    ) -> None:
        super().__init__(
            obs_dim,
            obs_len,
            state_dim,
            state_len,
            action_dim,
            machine_num,
            max_job_num,
            file_path,
            rng,
        )
        self.machine_num = 10
        self.job_file_path = file_path

    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_list[self.job_num].insert_time
        ):
            self.uncomplete_job.append(self.job_list[self.job_num])
            self.job_num += 1

    def reset(self):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        self.time_step, self.job_num = 0, 0
        self.count_actions = [0 for _ in range(self.action_dim)]
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.reward_calculator = AsyncTardinessReward(self.machine_num)
        jobs = None
        self.job_list = []
        with open(self.job_file_path, "r") as f:
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
        decision_machines = self.get_decsion_machines()
        self.current_machine = decision_machines[0]
        self.available_jobs = self.get_available_jobs()
        obs_i, obs_mask = self.get_obs_i()
        global_state = self.get_global_state()

        return obs_i, obs_mask, global_state
