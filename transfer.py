from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
import numpy as np
from scheduling_env.training_env import TrainingEnv
from scheduling_env.fjsp_eval_env import TRAN_ENV
from params import PARAMS
from scheduling_env.basic_scheduling_algorithms import PDR_RULES
MODEL_SAVE_PATH = "./models/num_25_50"
MODEL_FULL_PATH = MODEL_SAVE_PATH +".zip"


if __name__ == "__main__":
    result = []
    def make_env(i):
        def _init():
            return TRAN_ENV(
                state_dim=PARAMS["state_dim"],
                action_dim=PARAMS["action_dim"],
                machine_num=12,
                max_job_num = PARAMS["max_job_num"][i],
                lambda_rate = PARAMS["lambda_rate"][i],
                job_file_path = PARAMS["fjsp_same_path"],
                seed_list = PARAMS["seed_list"],
            )
        return _init
    methods = [method.__name__ for method in PDR_RULES.values()]
    methods = ['RL']+methods
    for i in range(len(PARAMS["lambda_rate"])):

        env_fns = [make_env(i)]
        env = DummyVecEnv(env_fns)
        env = VecNormalize(env, norm_obs=True, norm_reward=True ,training=False)
        env.training = False
        env.norm_reward = True
        env.norm_obs = True
        model = PPO.load(MODEL_FULL_PATH, env=env, device="cuda")
        all_episode_tardiness_sums = [
            [] for _ in range(len(PDR_RULES)+1)
        ]
        action_counts = []
        #评估model
        for episode in range(100):
            obs = env.reset()
            done = [False]
            while not done[0]:  # 假设 DummyVecEnv 中只有一个环境
                # 使用 deterministic=True 进行评估，以获得确定性行为
                action, _states = model.predict(obs, deterministic=True)

                # env.step() 返回的是一个元组，每个元素都是一个列表/数组 (对应每个子环境)
                obs, reward, done,  info = env.step(action)
                if done[0]:
                    all_episode_tardiness_sums[0].append(np.sum(info[0]["tardiness"]))
                    action_counts.append(info[0]['actions_count'])
                    # print(info[0]['seed'])

        # 测试PDR
        for pdr in range(len(PDR_RULES)):
            method = PDR_RULES[pdr]
            # print(f"Testing PDR: {method.__name__}")
            env = TRAN_ENV(
                state_dim=PARAMS["state_dim"],
                action_dim=PARAMS["action_dim"],
                machine_num=15,
                max_job_num = PARAMS["max_job_num"][i],
                lambda_rate = PARAMS["lambda_rate"][i],
                job_file_path = PARAMS["fjsp_same_path"],
                seed_list = PARAMS["seed_list"],
            )
            for episode in range(100):
                obs,info = env.reset()
                done = False
                while not done:
                    _, _, done,_, info = env.step_by_sr(pdr)
                    if done:
                        all_episode_tardiness_sums[pdr+1].append(np.sum(info["tardiness"]))
        result.append(np.mean(all_episode_tardiness_sums,axis=1))
    result = np.array(result)/np.array(PARAMS["max_job_num"]).reshape(-1,1)
    print(result)
    print(result.min(axis=1))
    print(np.sum(action_counts,axis=0))
    parameter_labels = [f"λ={PARAMS['lambda_rate'][i]}, jobs={PARAMS['max_job_num'][i]}" 
                       for i in range(len(PARAMS['max_job_num']))]
    
    # 创建DataFrame
    # 第一列是参数说明，后面的列是各个方法的结果
    data = {'Parameters': parameter_labels}
    
    # 添加每个方法的结果作为列
    for i, method_name in enumerate(methods):
        data[method_name] = result[:, i]
    import pandas as pd
    df = pd.DataFrame(data)
    
    # 保存到Excel
    df.to_excel('results.xlsx', index=False)
