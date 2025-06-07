import pandas as pd
import numpy as np
import pickle
from params import PARAMS
from scheduling_env.Instance_Generator import Instance_Generator
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO

# 方法列表
METHODS = ["RL"]
num_job = 10

# 初始化 PPO 并加载模型
ppo = PPO(
    local_state_dim=PARAMS["local_state_dim"],
    local_state_len=PARAMS["local_state_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    a_lr=PARAMS["actor_lr"],
    c_lr=PARAMS["critic_lr"],
    gamma=PARAMS["gamma"],
    lmbda=PARAMS["lmbda"],
    epochs=PARAMS["epochs"],
    eps=PARAMS["eps"],
    device=PARAMS["device"],
    weights=PARAMS["weights"],
    batch_size=PARAMS["batch_size"],
)

ppo.load_model(path="HFSD/models/ppo_model_RL.pth")

# 存储最终评估数据
result_records = []

def eval_model(env: TrainingEnv, method: str):
    obs = env.reset(1)
    done = False
    while not done:
        if method == "RL":
            action = ppo.take_action(obs)
            obs, _, done, _ = env.step(action)
    return env.compute_slack_time(), env.compute_machine_utiliaction(), env.time_step

def ev(data, machine_num, E_ave, new_insert):
    slack_time= []
    utilization = []
    makespan = []
    for i in range(num_job):
        for method in METHODS:
            env = TrainingEnv(
                PARAMS["action_dim"],
                machine_num,
                E_ave,
                new_insert,
                data[i]["job_info_list"],
                data[i]["arrival_times"],
            )
            t, u, m = eval_model(env, method)
            slack_time.append(t)
            utilization.append(u)
            makespan.append(m)
            print(f"{method:}\t {t:.2f}\t{u:.4f}\t{m:.2f}")

    result_records.append({
        "method": method,
        "job_num": new_insert,
        "E_ave": E_ave,
        "slack_time": np.mean(slack_time),
        "utilization": np.mean(utilization),
        "makespan": np.mean(makespan),
    })

if __name__ == "__main__":
    machine_num = [10,20,30]
    E_aves = [50, 100, 200]
    new_inserts = [20, 30, 40]

    for job_num in new_inserts:
        for E_ave in E_aves:
            with open(f"HFSD/job/{10}_{E_ave}_{job_num}.pkl", "rb") as f:
                data = pickle.load(f)
            ev(data, 10, E_ave, job_num)

    # 保存为 Excel 文件
    df = pd.DataFrame(result_records)
    df.to_excel("HFSD/evaluation_result.xlsx", index=False)
    print("评估结果已保存至 HFSD/evaluation_result+20.xlsx")
