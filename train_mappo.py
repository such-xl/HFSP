import os
import torch
import numpy as np
import json
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
from scheduling_env.training_env import TrainingEnv
from scheduling_env.MAPPO import AsyncMAPPO
from scheduling_env.basic_scheduling_algorithms import noname_2


def train_async_mappo(
    env, mappo, num_episodes=1000, batch_size=64, epochs=10, max_steps=200
):
    reward_history = []
    make_span_history = []
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/scheduling_env/data/train_data/"
    )
    job_name = "ela01.fjs"
    job_path = data_path + job_name
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": {},
    }
    for i in range(1, env._max_machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []
    for episode in range(num_episodes):
        G = {}
        for i in range(1, env._max_machine_num + 1):
            G[f"agent_{i}"] = 0
        obs_i = env.reset(job_path)
        global_state, state_mask = env.get_global_state()
        # done_n = [False] * mappo.n_agents
        episode_rewards = np.zeros(env._machine_num)
        done, truncated = False, False
        while not done and not truncated:

            active_agent_id = env._current_machine.id  # 获取当前需要决策的智能体

            action, log_prob, _ = mappo.select_action(obs_i, active_agent_id - 1)

            next_obs, reward, done, truncated = env.step(action)

            next_global_state, next_state_mask = env.get_global_state()
            G[f"agent_{active_agent_id}"] += reward
            mappo.store_experience(
                obs_i,
                action,
                reward,
                next_obs,
                True,  # done
                global_state,
                state_mask,
                next_global_state,
                next_state_mask,
                log_prob,
                active_agent_id - 1,
            )
            # obs_n[active_agent] = next_obs
            obs_i = next_obs
            global_state = next_global_state
            state_mask = next_state_mask
        machine = env._machines.head
        while machine:
            record["reward"][f"agent_{machine.id}"].append(G[f"agent_{machine.id}"])
            record["utilization_rate"][f"agent_{machine.id}"].append(
                machine.get_utilization_rate(env.time_step)
            )
            machine = machine.next
        record["makespan"][f"episode_{episode}"] = env.time_step

        # mappo.update_reward(reward)
        actor_loss, critic_loss, entropy = mappo.update(batch_size, epochs)
        make_span_history.append(env.time_step)

        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.4f}, Critic Loss {critic_loss:.4f}, make_span {env.time_step}"
        )
        reward_history.append(np.mean(episode_rewards))
    with open("record.json", "w") as f:
        json.dump(record, f)


def scheduling_algotithm(
    env,
    num_episodes=1000,
):
    data_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/scheduling_env/data/train_data/"
    )
    job_name = "Mk06.fjs"
    job_path = data_path + job_name
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": {},
    }
    for i in range(1, env._max_machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []
    for episode in range(num_episodes):
        G = {}
        reward = 0
        for i in range(1, env._max_machine_num + 1):
            G[f"agent_{i}"] = 0
        obs_i = env.reset(job_path)
        global_state, state_mask = env.get_global_state()
        # done_n = [False] * mappo.n_agents
        episode_rewards = np.zeros(env._machine_num)
        done, truncated = False, False
        while not done and not truncated:

            active_agent_id = env._current_machine.id  # 获取当前需要决策的智能体

            action = noname_2(env._job_list, env._current_machine, env.compute_UR())

            next_obs, reward, done, truncated = env.step_by_sr(action)

            next_global_state, next_state_mask = env.get_global_state()
            G[f"agent_{active_agent_id}"] += reward
        machine = env._machines.head
        while machine:
            record["reward"][f"agent_{machine.id}"].append(G[f"agent_{machine.id}"])
            record["utilization_rate"][f"agent_{machine.id}"].append(
                machine.get_utilization_rate(env.time_step)
            )
            machine = machine.next
        record["makespan"][f"episode_{episode}"] = env.time_step
        print(f"episode {episode+1}: makespan {env.time_step}")
    with open("record_sr.json", "w") as f:
        json.dump(record, f)


PARAMS = {
    "num_episodes": 2000,
    "batch_size": 64,
    "learning_rate": 6e-6,
    "gamma": 1,
    "tau": 0.005,
    "obs_dim": 6,
    "obs_len": 5,
    "global_state_dim": 6,
    "global_state_len": 10,
    "action_dim": 5,
    "max_machine_num": 20,
    "max_job_num": 10,
    "share_parameters": False,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
}

env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    max_machine_num=PARAMS["max_machine_num"],
    max_job_num=PARAMS["max_job_num"],
)

mappo = AsyncMAPPO(
    n_agents=PARAMS["max_machine_num"],
    obs_dim=PARAMS["obs_dim"],
    obs_len=PARAMS["obs_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    lr=PARAMS["learning_rate"],
    gamma=PARAMS["gamma"],
    share_parameters=PARAMS["share_parameters"],
    num_heads=PARAMS["num_heads"],
    device=PARAMS["device"],
)

train_async_mappo(
    env=env,
    mappo=mappo,
    num_episodes=PARAMS["num_episodes"],
    batch_size=PARAMS["batch_size"],
    epochs=10,
    max_steps=200,
)
# scheduling_algotithm(
#     env=env,
#     num_episodes=PARAMS["num_episodes"],
# )
