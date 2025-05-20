from stable_baselines3 import PPO,DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
import numpy as np
from scheduling_env.training_env import TrainingEnv
from params import PARAMS
from scheduling_env.basic_scheduling_algorithms import PDR_RULES
MODEL_SAVE_PATH = "./models/main"
MODEL_FULL_PATH = MODEL_SAVE_PATH +".zip"


if __name__ == "__main__":
    def make_env(i):
        def _init():
            return TrainingEnv(
                state_dim=PARAMS["state_dim"],
                action_dim=PARAMS["action_dim"],
                machine_num=PARAMS["machine_num"],
                max_job_num = PARAMS["max_job_num"][i],
                lambda_rate = PARAMS["lambda_rate"][i],
                job_file_path = PARAMS["fjsp_same_path"],
                seed_list = PARAMS["seed_list"],
            )
        return _init
    env_fns = [make_env(0)]
    
    env = DummyVecEnv(env_fns)
    env = VecNormalize.load(MODEL_SAVE_PATH+".pkl", env)
    env.training = False
    # env.norm_reward = False
    try:
        model = PPO.load(MODEL_FULL_PATH, env=env, device="cuda")  # 或 "cpu"
        print(f"模型 {MODEL_FULL_PATH} 加载成功.")
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()



    # --- 现在你可以运行你的自定义评估循环 ---
    print("\n开始自定义评估循环...")

    # 评估多少个 episodes
    num_episodes_to_evaluate = 1  # 例如评估5个episodes
    all_episode_actions_counts = []
    all_episode_tardiness_sums = [
        [] for _ in range(len(PDR_RULES)+1)
    ]
    all_episode_rewards = []

    obs = env.reset()
    for i in range(num_episodes_to_evaluate):
        done = [False]
        episode_reward_sum = 0
        episode_length = 0

        while not done[0]:  # 假设 DummyVecEnv 中只有一个环境
            # 使用 deterministic=True 进行评估，以获得确定性行为
            action, _states = model.predict(obs, deterministic=True)

            # env.step() 返回的是一个元组，每个元素都是一个列表/数组 (对应每个子环境)
            obs, reward, done,  info = env.step(action)
            if done[0]:
                all_episode_tardiness_sums[0].append(np.sum(info[0]["tardiness"]))
                all_episode_actions_counts.append(info[0]['actions_count'])
                print(np.sum(info[0]['actions_count']))
            episode_reward_sum += reward
            episode_length += 1

    # 测试PDR
    for pdr in range(len(PDR_RULES)):
        method = PDR_RULES[pdr]
        print(f"Testing PDR: {method.__name__}")
        env = TrainingEnv(
            state_dim=PARAMS["state_dim"],
            action_dim=PARAMS["action_dim"],
            machine_num=PARAMS["machine_num"],
            max_job_num=PARAMS["max_job_num"][0],
            lambda_rate=PARAMS["lambda_rate"][0],
            job_file_path=PARAMS["fjsp_same_path"],
            seed_list=PARAMS["seed_list"],
        )
        env.reset()
        for eps in range(num_episodes_to_evaluate):
            obs,info = env.reset()
            done = False
            while not done:
                _, _, done,_, info = env.step_by_sr(pdr)
                if done:
                    all_episode_tardiness_sums[pdr+1].append(np.sum(info["tardiness"]))
                    # print(info['seed'])
    print(np.mean(all_episode_tardiness_sums,axis=1))
    # print(np.std(all_episode_tardiness_sums, axis=1))

