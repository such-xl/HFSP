import os
import torch
import numpy as np
import json

from scheduling_env.training_env import TrainingEnv
from scheduling_env.MAPPO import AsyncMAPPO

from scheduling_env.basic_scheduling_algorithms import noname_2
from scheduling_env.utils import ExponentialTempScheduler


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
        "makespan": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []
    temp_scheduler = ExponentialTempScheduler(
        initial_temp=1.0, min_temp=0.01, decay_rate=0.995
    )

    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0

        current_temp = temp_scheduler.step()
        obs_i, obs_mask, global_state, state_mask = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action, log_prob, _ = mappo.select_action(
                obs_i,
                obs_mask,
                tau=current_temp,
                hard=(current_temp < 0.5),
            )
            (
                next_obs_i,
                next_obs_mask,
                next_global_state,
                next_state_mask,
                reward,
                done,
                truncated,
            ) = env.step(action)
            # print(f"actino:{action} reward:{reward}")
            G[f"agent_{env.current_machine.id}"] += reward
            mappo.store_experience(
                obs_i,
                obs_mask,
                action,
                reward,
                next_obs_i,
                next_obs_mask,
                True,
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
        record["makespan"].append(env.time_step)
        actor_loss, critic_loss, entropy = mappo.update(batch_size, epochs)
        record["actor_loss"].append(actor_loss)
        record["critic_loss"].append(critic_loss)
        record["entropy"].append(entropy)

        print(
            f"Episode {episode + 1}/{num_episodes}: Actor Loss {actor_loss:.4f}, Critic Loss {critic_loss:.4f}, make_span {env.time_step}, avg_reward {np.mean(list(G.values())):.4f}, tau {current_temp:.4f},  entropy:{entropy:.4f}idle_action:{env.idle_action}"
        )
    with open("result/record_rl.json", "w") as f:
        json.dump(record, f)


def sr(
    env: TrainingEnv,
    num_episodes=1000,
):
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": [],
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []

    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0
        _, _, _, _ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = noname_2(env.available_jobs, env.current_machine, env.compute_UR())
            reward, done, truncated = env.step_by_sr(action)
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
        record["makespan"].append(env.time_step)

        print(f"Episode {episode + 1}/{num_episodes}: make_span {env.time_step}")

    with open("result/record_sr.json", "w") as f:
        json.dump(record, f)


PARAMS = {
    "num_episodes": 800,
    "batch_size": 32,
    "actor_lr": 6e-5,
    "critic_lr": 3e-4,
    "gamma": 1,
    "obs_dim": 6,
    "obs_len": 6,
    "global_state_dim": 6,
    "global_state_len": 30,
    "action_dim": 6,
    "max_machine_num": 20,
    "max_job_num": 10,
    "share_parameters": False,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__))
    + "/scheduling_env/data/train_data/",
    "job_name": "vla20.fjs",
    "train": True,
}

if PARAMS["train"]:

    env = TrainingEnv(
        obs_dim=PARAMS["obs_dim"],
        obs_len=PARAMS["obs_len"],
        state_dim=PARAMS["global_state_dim"],
        state_len=PARAMS["global_state_len"],
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
        actor_lr=PARAMS["actor_lr"],
        critic_lr=PARAMS["critic_lr"],
        gamma=PARAMS["gamma"],
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
else:
    env_sr = TrainingEnv(
        obs_dim=PARAMS["obs_dim"],
        obs_len=PARAMS["obs_len"],
        state_dim=PARAMS["global_state_dim"],
        state_len=PARAMS["global_state_len"],
        action_dim=PARAMS["action_dim"],
        max_job_num=PARAMS["max_job_num"],
        job_file_path=PARAMS["data_path"] + PARAMS["job_name"],
    )
    sr(
        env=env_sr,
        num_episodes=PARAMS["num_episodes"],
    )
