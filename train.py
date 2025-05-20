import os
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv,VecMonitor
from callback import EpisodeMetricsCallback
import numpy as np
from scheduling_env.training_env import TrainingEnv
from params import PARAMS
from scheduling_env.network import CustomFeatureExtractor
policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
}

if __name__ == "__main__":

    def make_env(i):
        def _init():
            return Monitor(TrainingEnv(
                state_dim=PARAMS["state_dim"],
                action_dim=PARAMS["action_dim"],
                machine_num=PARAMS["machine_num"],
                max_job_num = PARAMS["max_job_num"][i],
                lambda_rate = PARAMS["lambda_rate"][i],
                job_file_path = PARAMS["fjsp_same_path"],
                seed_list = PARAMS["seed_list"],
            ))
        return _init
    env_fns = [make_env(i) for i in range(3)]
    
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, verbose=1,device="cuda", tensorboard_log="./ppo_logs/",learning_rate=3e-5)
    # model = DQN("MlpPolicy",env,device="cuda",policy_kwargs=policy_kwargs,tensorboard_log="./ppo_logs/",verbose=1)
    callback = EpisodeMetricsCallback(log_path="./ppo_logs/main.csv")
    model.learn(total_timesteps=6_000_000, callback=callback)
    model.save("models/main")
    env.save("models/main.pkl")
