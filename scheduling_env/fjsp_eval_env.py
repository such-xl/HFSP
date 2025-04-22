import numpy as np
from .training_env import TrainingEnv

np.random.seed(42)


class FJSP_EVAL_ENV(TrainingEnv):
    def __init__(
        self,
        obs_dim,
        obs_len,
        state_dim,
        state_len,
        action_dim,
        machine_num,
        max_job_num,
        job_file_path,
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
            job_file_path,
            rng,
        )

    def renew_job_data(self):
        self.job_arrivals = super().create_job_arriavl_seq()

    def reset(self):
        # self.renew_job_data()
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
