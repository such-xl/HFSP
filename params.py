import os
import torch

PARAMS = {
    "num_episodes": 800,
    "batch_size": 12,
    "actor_lr": 3e-3,
    "critic_lr": 3e-4,
    "gamma": 0.98,
    "obs_dim": 8,
    "obs_len": 5,
    "global_state_dim": 6,
    "global_state_len": 48,
    "action_dim": 4,
    "max_job_num": 200,
    "share_parameters": False,
    "machine_num": 10,
    "num_heads": 6,
    "device": (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    "data_path": os.path.dirname(os.path.abspath(__file__)) + "/experiment/fjsp/",
    "job_name": "fjsp_same.json",
    "train": False,
    "idle_action": False,
    "model_path": "models/",
}
