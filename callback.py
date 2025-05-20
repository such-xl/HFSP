# callback.py
import csv
import os
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, log_path="./episode_metrics.csv", verbose=0):
        super(EpisodeMetricsCallback, self).__init__(verbose)
        self.log_path = log_path
        self.header_written = False

    def _on_training_start(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps","lambda_rate","episode_reward", "tardiness"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "tardiness" in info and "episode_reward" in info:
                with open(self.log_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [self.num_timesteps,info["lambad_rate"],info["episode_reward"], sum(info["tardiness"])]
                    )
        return True
