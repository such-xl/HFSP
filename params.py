import os
import torch
PARAMS = {
"num_episodes": 500,
    "batch_size": 32,
    "actor_lr": 3e-4,
    "critic_lr": 3e-5,
    "local_state_dim": 6,
    "local_state_len": 5,
    "gamma": 0.98,
    "tau": 0.005,
    "global_state_dim": 6,
    "global_state_len": 15,
    "action_dim": 4,
    "machine_num": 10,
    "E_ave":20,
    "new_insert": 10,
    "lmbda":0.95,
    "eps":0.3,
    "epochs":20,
    "weights":[0.7,0.3],
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__))
    + "/experiment/fjsp/",
    "job_name": "fjsp_same_test.json",
    "train": False,
    "model_path": "models/",
}