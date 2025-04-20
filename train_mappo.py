import os
import torch
import numpy as np
import json
import time
from scheduling_env.training_env import TrainingEnv
from scheduling_env.MAPPO import AsyncMAPPO

from scheduling_env.basic_scheduling_algorithms import EDD,MS,SRO,CR,noname_2
from scheduling_env.utils import ExponentialTempScheduler

def train_async_mappo(
    env: TrainingEnv,
    mappo: AsyncMAPPO,
    num_episodes=2000,
    batch_size=8,
    epochs=10,
    output_path="default",
):
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "wait_time": {},
        "tardiness":{},
    }
    system_record = {
        "tardiness_rate":[],
        "avg_tardiness":[],
        "system_reward":[],
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []
    for i in range(1, 300 + 1):
        record["wait_time"][f"job_{i}"] = []
        record["tardiness"][f"job_{i}"] = []
    temp_scheduler = ExponentialTempScheduler(
        initial_temp=5.0, min_temp=0.01, decay_rate=0.995
    )
    current_temp = 0
    action = None
    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0
        current_temp = temp_scheduler.step()
        obs_i, obs_mask, global_state = env.reset()
        done, truncated = False, False
        action_count = [0 for _ in range(env.action_dim)]
        while not done and not truncated:
            action, log_prob, _ = mappo.select_action(
                obs_i,
                obs_mask,
                tau=current_temp,
                hard=(current_temp < 0.5),
            )
            action_count[action] += 1
            (
                next_obs_i,
                next_obs_mask,
                next_global_state,
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
                True,
                global_state,
                next_global_state,
                log_prob,
                env.current_machine.id - 1,
            )
            obs_i = next_obs_i
            obs_mask = next_obs_mask
            global_state = next_global_state
        
        # mappo.update_last_reward(env.rewards)
        # for i,r in enumerate(env.rewards):
        #     G[f"agent_{i+1}"] += r
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
        actor_loss, critic_loss, entropy, kl_div = mappo.update(batch_size, epochs,tau=current_temp,
                hard=(current_temp < 0.5),)
        record["actor_loss"].append(actor_loss)
        record["critic_loss"].append(critic_loss)
        record["entropy"].append(entropy)
        tard_sum ,tard_count  = 0,0
        job = env.complete_job.head
        while job:
            record["wait_time"][f"job_{job.id}"].append(job.wait_time)
            record["tardiness"][f"job_{job.id}"].append(job.tard_time)
            tard_sum += job.tard_time
            if job.tard_time > 0:
                tard_count += 1
            job = job.next
        
        print(
            f"Episode {episode + 1}/{num_episodes}: Actor Loss {actor_loss:.4f}, Critic Loss {critic_loss:.4f},KL {kl_div:.4f}, make_span {env.time_step}, avg_reward {np.mean(list(G.values())):.4f}, tau {current_temp:.4f},  entropy:{entropy:.4f}, tard_sum:{tard_sum/tard_count:.4f} action_count:{action_count},no_repete:{env.count_actions}"
        )
        _,system_details = env.reward_calculator.calculate_system_reward(env.time_step)
        # system_record["system_mean_utilization"].append(system_details["system_mean_utilization"])
        # system_record["system_std_utilization"].append(system_details["system_std_utilization"])
        # system_record["system_reward"].append(system_details["system_reward"])
        print(system_details["avg_tardiness"]*system_details["tardiness_rate"] * 100)
        # print(env.reward_calculator.system_history)
        system_record["tardiness_rate"].append(system_details["tardiness_rate"])
        system_record["avg_tardiness"].append(system_details["avg_tardiness"])
        system_record["system_reward"].append(system_details["system_reward"])
    with open(f"result/record_{output_path}_rl.json", "w") as f:
        json.dump(record, f)
    with open(f"result/record_{output_path}_system.json", "w") as f:
        json.dump(system_record, f)
    mappo.save_model()

def sr(env: TrainingEnv, num_episodes=1000, output_path="default"):
    record = {
        "reward": {},
        "utilization_rate": {},
        "makespan": [],
        "wait_time": {},
    }
    for i in range(1, env.machine_num + 1):
        record["reward"][f"agent_{i}"] = []
        record["utilization_rate"][f"agent_{i}"] = []
    for i in range(1, 300 + 1):
        record["wait_time"][f"job_{i}"] = []
    for episode in range(num_episodes):
        G = {}
        for i in range(1, env.machine_num + 1):
            G[f"agent_{i}"] = 0
        _, _, _ = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            # action = (env.available_jobs, env.current_machine, env.compute_UR())
            # action1 = CR(env.available_jobs, env.time_step)
            # action = noname_2(env.available_jobs, env.current_machine, env.compute_UR())
            action = EDD(env.available_jobs)[0]
            # action3 = MS(env.available_jobs,env.time_step)
            # action4 = SRO(env.available_jobs,env.time_step)
            # action = np.random.choice([action1,action2,action3,action4])
            reward, done, truncated = env.step_by_sr(action)
            G[f"agent_{env.current_machine.id}"] += reward

        list(
            record["reward"][f"agent_{machine.id}"].append(G[f"agent_{machine.id}"])
            for machine in env.machines
        )
        # print([(x[0]["type"],x[1]) for x in env.job_arrivals])
        list(
            record["utilization_rate"][f"agent_{machine.id}"].append(
                machine.get_utilization_rate(env.time_step)
            )
            for machine in env.machines
        )
        record["makespan"].append(env.time_step)
        job = env.complete_job.head
        tard_sum = 0
        while job:
            record["wait_time"][f"job_{job.id}"].append(job.wait_time)
            tard_sum += job.tard_time
            job = job.next
        print(f"Episode {episode + 1}/{num_episodes}: make_span {env.time_step}, tard_sum: {tard_sum}")

    with open(f"result/record_{output_path}_sr.json", "w") as f:
        json.dump(record, f)


PARAMS = {
    "num_episodes":1000,
    "batch_size": 12,
    "actor_lr": 3e-4,
    "critic_lr": 3e-5,
    "gamma": 0.98,
    "obs_dim": 8,
    "obs_len": 5,
    "global_state_dim": 6,
    "global_state_len": 48,
    "action_dim": 4,
    "max_machine_num": 10,
    "max_job_num": 100,
    "share_parameters": False,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__))
    + "/experiment/fjsp/",
    "job_name": "record.json",
    "train": False,
    "idle_action": False,
    "model_path": "models/",
}

if __name__ == "__main__":
    """
    1. fjs                      sla39.fjs
    2. 加工时间不同，fjsp        Mk10.fjs
    3. 加工时间相同，fjsp        vla39.fjs
    4. 加工时间不同，cfjsp
    5. 加工时间相同，cfjsp
    """
    task_type = {
        # "jsp": "jsp.json",
        # "fjsp_diff": "fjsp_diff.json",
        "fjsp_same": "fjsp_same.json",
        # "cfjsp_diff": "cfjsp_diff.json",
        # "cfjsp_same": "cfjsp_same.json",
    }
    for k, v in task_type.items():
        PARAMS["job_name"] = v
        # if not PARAMS["idle_action"]:
        #     PARAMS["action_dim"] = PARAMS["action_dim"] - 1
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
            model_save_path=PARAMS["model_path"]+"fjsp_same.pth",
        )
        train_async_mappo(
            env=env,
            mappo=mappo,
            num_episodes=PARAMS["num_episodes"],
            batch_size=PARAMS["batch_size"],
            epochs=10,
            output_path=k,
        )

        env_sr = TrainingEnv(
            obs_dim=PARAMS["obs_dim"],
            obs_len=PARAMS["obs_len"],
            state_dim=PARAMS["global_state_dim"],
            state_len=PARAMS["global_state_len"],
            action_dim=PARAMS["action_dim"],
            max_job_num=PARAMS["max_job_num"],
            job_file_path=PARAMS["data_path"] + PARAMS["job_name"],
        )
        sr(env=env_sr, num_episodes=PARAMS["num_episodes"], output_path=k)
