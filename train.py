import os
from stable_baselines3 import PPO,DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv,VecMonitor
from callback import EpisodeMetricsCallback,ActionStatsOnlyCallback
import numpy as np
from scheduling_env.training_env import TrainingEnv
from params import PARAMS
import torch
custom_policy_kwargs = {
    "net_arch": [128,128,64],  # 3层：128, 128, 64
    "activation_fn": torch.nn.LeakyReLU
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
    env_fns = [make_env() for i in range(1)]
    
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, verbose=0,device="cuda",policy_kwargs=custom_policy_kwargs, tensorboard_log="./ppo_logs/",
                clip_range=0.1,n_steps=2048,batch_size=128,ent_coef=0.001,learning_rate=1e-4,gamma=0.99,target_kl=0.02)
    # model = DQN("MlpPolicy",env,device="cuda",policy_kwargs=policy_kwargs,tensorboard_log="./ppo_logs/",verbose=1)
    callback1 = EpisodeMetricsCallback(log_path="./ppo_logs/main_new3.csv")
    callback2 = ActionStatsOnlyCallback(action_space_size=PARAMS["action_dim"])
    model.learn(total_timesteps=2_000_000, callback=[callback1,callback2],progress_bar=True)
    model.save("models/main_new3")
    env.save("models/main_new3.pkl")
