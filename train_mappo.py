import os
import torch
import numpy as np
import json

from scheduling_env.training_env import TrainingEnv
from scheduling_env.MAPPO import AsyncMAPPO

from scheduling_env.basic_scheduling_algorithms import noname_2


def train_async_mappo(
    env: TrainingEnv,
    mappo: AsyncMAPPO,
    num_episodes=1000,
    batch_size=64,
    epochs=10,
):
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": {},
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []

    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0
        obs_i, obs_mask, global_state, state_mask = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action, log_prob, _ = mappo.select_action(obs_i,obs_mask,env.current_machine.id - 1)
            (
                next_obs_i,
                next_obs_mask,
                next_global_state,
                next_state_mask,
                reward,
                done,
                truncated,
            ) = env.step(action)

            G[f"agent_{env.current_machine.id}"] += reward
            mappo.store_experience(
                obs_i,
                obs_mask,
                action,
                reward,
                next_obs_i,
                next_obs_mask,
                done if mappo.share_parameters else True,# done
                global_state,
                state_mask,
                next_global_state,
                next_state_mask,
                log_prob,
                env.current_machine.id - 1,
            )
            obs_i = next_obs_i
            obs_mask = next_obs_mask
            global_state = next_global_state
            state_mask = next_state_mask

        list(
            record["reward"][f"agent_{machine.id}"].append(G[f"agent_{machine.id}"])
            for machine in env.machines
        )

        list(
            record["utilization_rate"][f"agent_{machine.id}"].append(
                machine.get_utilization_rate(env.time_step)
            )
            for machine in env.machines
        )
        record["makespan"][f"episode_{episode}"] = env.time_step

        # mappo.update_reward(reward)
        actor_loss, critic_loss, entropy = mappo.update(batch_size, epochs)

        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.4f}, Critic Loss {critic_loss:.4f}, make_span {env.time_step}"
        )

    with open("record.json", "w") as f:
        json.dump(record, f)


def sr(
    env: TrainingEnv,
    num_episodes=1000,
):
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": {},
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []

    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0
        _,_,_,_ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = noname_2(env.available_jobs,env.current_machine,env.compute_UR())
            reward,done,truncated = env.step_by_sr(action)
            G[f"agent_{env.current_machine.id}"] += reward

        list(
            record["reward"][f"agent_{machine.id}"].append(G[f"agent_{machine.id}"])
            for machine in env.machines
        )

        list(
            record["utilization_rate"][f"agent_{machine.id}"].append(
                machine.get_utilization_rate(env.time_step)
            )
            for machine in env.machines
        )
        record["makespan"][f"episode_{episode}"] = env.time_step


        print(
            f"Episode {episode + 1}/{num_episodes}: make_span {env.time_step}"
        )

    with open("record.json", "w") as f:
        json.dump(record, f)

PARAMS = {
    "num_episodes": 2000,
    "batch_size": 32,
    "learning_rate": 6e-6,
    "gamma": 1,
    "obs_dim": 6,
    "obs_len": 6,
    "global_state_dim": 6,
    "global_state_len": 30,
    "action_dim": 6,
    "max_machine_num": 20,
    "max_job_num": 30,
    "share_parameters": True,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__))
    + "/scheduling_env/data/train_data/",
    "job_name": "Mk06.fjs",
}

env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    max_job_num=PARAMS["max_job_num"],
    job_file_path=PARAMS["data_path"] + PARAMS["job_name"],
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
)
# sr(
#     env=env,
#     num_episodes=PARAMS["num_episodes"],
# )
