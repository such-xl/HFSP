import numpy as np
import json
from params import PARAMS
from scheduling_env.Instance_Generator import Instance_Generator
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO
np.random.seed(42)
EVAL_EPISODE = 10

machine_num = PARAMS['machine_num']
E_ave = PARAMS['E_ave']
new_insert = PARAMS['new_insert']
METHODS = ["RL", "SPT", "LPT", "LRPT","SRPT","Random"]

ppo = PPO(
    local_state_dim = PARAMS["local_state_dim"],
    local_state_len = PARAMS["local_state_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    a_lr=PARAMS["actor_lr"],
    c_lr=PARAMS["critic_lr"],
    gamma=PARAMS["gamma"],
    lmbda=PARAMS["lmbda"],
    epochs =PARAMS["epochs"],
    eps=PARAMS["eps"],
    device=PARAMS["device"],
    weights=PARAMS["weights"],
    batch_size = PARAMS["batch_size"],
)
ppo.load_model(path = f"HFSD/models/ppo_model_{machine_num}_{E_ave}_{new_insert}_RL_first_m_u.pth")
record = {method: {"tardiness": [], "U": [],"makespan":[]} for method in METHODS}
def eval_model(env:TrainingEnv,method:str):
    for episode in range(1):
        obs = env.reset()
        done,truncted = False,False
        while not (done or truncted):
            if method == "RL":
                action = ppo.take_action(obs)
                next_obs,_,done, truncted = env.step(action)
                obs = next_obs
            elif method == "SPT":
                done, truncted = env.sr(0)
            elif method == "LPT":
                done, truncted = env.sr(1)
            elif method == "LRPT":
                done, truncted = env.sr(2)
            elif method == "SRPT":
                done, truncted = env.sr(3)
            elif method == "Random":
               done, truncted = env.sr(4)
            else:
                raise ValueError("method not found")
        record[method]["tardiness"].append(env.sum_tardiness())
        record[method]["U"].append(env.compute_machine_utiliaction())
        record[method]["makespan"].append(env.time_step)

if __name__ == "__main__":
    jobs = []
    arrivals = []
    for i in range(1):
        jobs_list, arrivals_list = Instance_Generator(
            M_num=PARAMS["machine_num"],
            E_ave=PARAMS["E_ave"],
            New_insert=PARAMS["new_insert"]
            )
        jobs.append(jobs_list)
        arrivals.append(arrivals_list)
    for i in range(1):
        # 为每个方法都用这同一批 jobs
        ENV = [
            TrainingEnv(
                action_dim=PARAMS["action_dim"],
                machine_num=PARAMS["machine_num"],
                E_ave=PARAMS["E_ave"],
                new_insert=PARAMS['new_insert'],
                job_info_list=jobs[i],
                job_arrival_time=arrivals[i]
            )
            for _ in METHODS
        ]
        for method,env in zip(METHODS,ENV):
                eval_model(env,method)
        
        print(f"\ttard\tu_m\tms")
        for method in METHODS:
            tard = record[method]["tardiness"][i]
            U_M = 1/record[method]["U"][i]
            makespan = record[method]["makespan"][i]
            # print(tard,U_M,makespan)[0]
            print(f"{method:}\t {tard:.2f}\t{U_M:.4f}\t{makespan:.2f}")
        with open(f'HFSD/results/{machine_num}_{E_ave}_{new_insert}_ddxl.json',"w") as f:
            json.dump(record,f)


        