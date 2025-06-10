import numpy as np
from .training_env import TrainingEnv
from .machine import Machine
from .job import Job,JobList
from .reward_2 import AsyncTardinessReward
np.random.seed(42)


class FJSP_EVAL_ENV(TrainingEnv):
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
        super().__init__(
            state_dim,
            action_dim,
            machine_num,
            max_job_num,
            lambda_rate,
            job_file_path,
            seed_list
        )
        self.rng2 = None
    def renew_job_data(self):
        self.job_arrivals = super().create_job_arriavl_seq(self.job_arrivals)

    def reset(self):
        self.rng2 = np.random.RandomState(self.seed_list[self.episode])
        return super().reset()

    def compute_slack_time(self):
        slack_time = 0
        job = self.complete_job.head
        while job:
            slack_time += job.wait_time
            job = job.next
        return slack_time

    def compute_tard_time(self):
        tard_time = 0
        job = self.complete_job.head
        while job:
            tard_time += job.tard_time
            job = job.next
        return tard_time

class TRAN_ENV(TrainingEnv):
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
        super().__init__(
            state_dim,
            action_dim,
            machine_num,
            max_job_num,
            lambda_rate,
            job_file_path,
            seed_list,
        )
    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_arrivals[self.job_num][1]
        ):
            job_info = self.job_arrivals[self.job_num][0]
            process_list = job_info["process_list"]
            # 为每道工序添加1-3个可用机器
            new_machines = [i for i in range(11,self.machine_num+1)]
            for process in process_list:
                num_to_add = self.rng2.randint(1,3)
                process_time = list(process.values())[0]
                available_machines = self.rng2.choice(
                    new_machines, num_to_add, replace=False
                )
                for m in available_machines:
                    process[m] = process_time

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
    def reset(self, seed=None, options=None):
        self.time_step, self.job_num = 0, 0
        self.episode_reward = 0
        self.eps_num += 1
        self.count_actions = [0 for _ in range(self.action_dim)]
        self.rng = np.random.RandomState(self.seed_list[self.eps_num])
        self.rng2 = np.random.RandomState(self.seed_list[self.eps_num])
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
