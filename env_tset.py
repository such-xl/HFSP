import copy
import numpy as np
import json
from params import PARAMS
from scheduling_env.fjsp_eval_env import FJSP_EVAL_ENV
from create_job import create_cfjsp_same_job
from scheduling_env.MAPPO import AsyncMAPPO
from scheduling_env.basic_scheduling_algorithms import noname_2, EDD, MS, SRO, CR, ATC

np.random.seed(42)
EVAL_EPISODE = 1

METHODS = ["RL", "EDD", "MS", "SRO", "CR", "ATC"]

mappo = AsyncMAPPO(
    n_agents=PARAMS["machine_num"],
    obs_dim=PARAMS["obs_dim"],
    obs_len=PARAMS["obs_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    actor_lr=PARAMS["actor_lr"],
    critic_lr=PARAMS["critic_lr"],
    gamma=PARAMS["gamma"],
    num_heads=PARAMS["num_heads"],
    device=PARAMS["device"],
    model_save_path=PARAMS["model_path"] + "fjsp_same.pth",
)
mappo.load_model(model_path="./models/fjsp_same.pth")

record = {
    method: {"tard": [], "slack": [], "util": [], "makespan": []} for method in METHODS
}

actions_count = [0 for _ in range(4)]


def eval_model(env: FJSP_EVAL_ENV, method: str):
    mean, std, tard, makespan = 0, 0, 0, 0
    for episode in range(1):
        obs_i, obs_mask, _ = env.reset()
        done, truncted = False, False
        while not (done or truncted):
            if method == "RL":
                action, _, _ = mappo.select_action(obs_i, obs_mask, eval_mode=True)
                actions_count[action] += 1
                next_obs, next_obs_mask, _, _, done, truncted = env.step(action)
                obs_i = next_obs
                obs_mask = next_obs_mask
            elif method == "EDD":
                action = EDD(env.available_jobs)[0]
                done, truncted = env.step_by_sr(action)
            elif method == "SRO":
                action = SRO(env.available_jobs, env.time_step)[0]
                done, truncted = env.step_by_sr(action)
            elif method == "MS":
                action = MS(env.available_jobs, env.time_step)[0]
                done, truncted = env.step_by_sr(action)
            elif method == "CR":
                action = CR(env.available_jobs, env.time_step)[0]
                done, truncted = env.step_by_sr(action)
            elif method == "ATC":
                action = ATC(env.available_jobs, env.time_step)[0]
                done, truncted = env.step_by_sr(action)
            else:
                raise ValueError("method not found")
        # End of while loop
        record[method]["tard"].append(env.compute_tard_time())
        record[method]["slack"].append(env.compute_slack_time())
        record[method]["util"].append(env.compute_UR())
        record[method]["makespan"].append(env.time_step)
    # print(f"{method}:\t{mean/EVAL_EPISODE} \t{std/EVAL_EPISODE} \t{tard/EVAL_EPISODE}\t{makespan/EVAL_EPISODE}")
    # print(f"{method}:\t{np.mean(env.compute_UR())*100:.4f} \t{np.std(env.compute_UR()):.4f} \t {env.time_step} \t {env.compute_tard_time()}")


if __name__ == "__main__":
    for episode in range(EVAL_EPISODE):
        seed = np.random.randint(1e5)
        rngs = [np.random.default_rng(seed) for _ in range(len(METHODS))]
        ENV = [
            FJSP_EVAL_ENV(
                obs_dim=PARAMS["obs_dim"],
                obs_len=PARAMS["obs_len"],
                state_dim=PARAMS["global_state_dim"],
                state_len=PARAMS["global_state_len"],
                action_dim=PARAMS["action_dim"],
                machine_num=PARAMS["machine_num"],
                max_job_num=PARAMS["max_job_num"],
                job_file_path=PARAMS["data_path"] + PARAMS["job_name"],
                rng=rngs[i],
            )
            for i in range(len(METHODS))
        ]
        for method, env in zip(METHODS, ENV):
            eval_model(env, method)

    print(f"\ttard\tu_m\tu_s\tslack\tms")
    for method in METHODS:
        tard = np.mean(record[method]["tard"])
        slack = np.mean(record[method]["slack"])
        util = np.array(record[method]["util"])
        util_mean = util.mean()
        util_std = np.mean(util.std(axis=1))
        makespan = np.mean(record[method]["makespan"])
        print(
            f"{method:}\t {tard:.2f}\t{util_mean:.4f}\t{util_std:.4f}\t{slack:.2f}\t{makespan:.2f}"
        )
    print(actions_count)
    with open("ddxl.json", "w") as f:
        json.dump(record, f)
