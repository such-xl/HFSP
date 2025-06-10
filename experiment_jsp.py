import os
import numpy as np
import natsort
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from scheduling_env.eval_env import JSPEvalEnv
from params import PARAMS

MODEL_SAVE_PATH = "./models/num_25_50"
MODEL_FULL_PATH = MODEL_SAVE_PATH + ".zip"

if __name__ == "__main__":

    def make_env(i):
        def _init():
            return JSPEvalEnv(
                state_dim=PARAMS["state_dim"],
                action_dim=PARAMS["action_dim"],
                machine_num=PARAMS["machine_num"],
                max_job_num = PARAMS["max_job_num"],
                seed_list = PARAMS["seed_list"],
                ur = [70,80,90],
                data_path = "./experiment/jsp/job_data/"
            )
        return _init
    env_fns = [make_env(0)]
    
    env = DummyVecEnv(env_fns)
    env = VecNormalize.load(MODEL_SAVE_PATH+".pkl", env)
    env.training = False
    env.norm_obs = False

    try:
        model = PPO.load(MODEL_FULL_PATH, env=env, device="cuda")  # 或 "cpu"
        print(f"模型 {MODEL_FULL_PATH} 加载成功.")
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()
    tards_records = {}
    for ur in [70,80,90]:
        tards_records[ur] = []
    obs = env.reset()
    for eps in range(300):
        print("===")
        print("eps",eps)
        done = [False]
        while not done[0]:
            action,_state = model.predict(obs,deterministic=True)
            obs,reward,done,info = env.step(action)
            if done[0]:
                tards_records[70+(eps//100)*10].append(info[0]['tardiness'])
    with open("experiment/jsp/resurt_all.json", "w") as f:
        json.dump(tards_records, f)
