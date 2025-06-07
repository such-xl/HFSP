import pandas as pd
import numpy as np
import pickle
from params import PARAMS
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO

METHODS = ["RL"]
num_job = 10

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

def eval_model(env: TrainingEnv, method: str):
    obs = env.reset(1)
    done = False
    while not done:
        if method == "RL":
            action = ppo.take_action(obs)
            obs, _, done, _ = env.step(action)
    return env.compute_slack_time(), env.compute_machine_utiliaction(), env.time_step

def evaluate_all_machines(E_ave, job_num, machine_list):
    """ 针对一个 (E_ave, job_num) 评估多个机器数，并统一收集结果 """
    result_dict = {
        "method": "RL",
        "job_num": job_num,
        "E_ave": E_ave
    }

    for machine_num in machine_list:
        file_path = f"HFSD/8_12_16/{machine_num}_{E_ave}_{job_num}.pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        slack_time = []
        utilization = []
        makespan = []

        for i in range(num_job):
            env = TrainingEnv(
                PARAMS["action_dim"],
                machine_num,
                E_ave,
                job_num,
                data[i]["job_info_list"],
                data[i]["arrival_times"],
            )
            t, u, m = eval_model(env, "RL")
            slack_time.append(t)
            utilization.append(u)
            makespan.append(m)
            print(t,u,m)

        # 添加横向列
        result_dict[f"slack_time_{machine_num}"] = np.mean(slack_time)
        result_dict[f"util_{machine_num}"] = np.mean(utilization)
        result_dict[f"makespan_{machine_num}"] = np.mean(makespan)

    return result_dict

if __name__ == "__main__":
    machine_list = [8, 12, 16]
    E_aves = [50, 100, 200]
    new_inserts = [20, 30, 40]

    result_records = []
    for job_num in new_inserts:
        for E_ave in E_aves:
            result = evaluate_all_machines(E_ave, job_num, machine_list)
            result_records.append(result)

    # 保存为 Excel 表格
    df = pd.DataFrame(result_records)
    df.to_excel("HFSD/evaluation_machines.xlsx", index=False)
    print("✅ 汇总结果已保存至 HFSD/evaluation_machines.xlsx")