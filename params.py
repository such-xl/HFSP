import os
import numpy as np

PARAMS = {
    "state_dim":50,
    "action_dim" : 5,
    "machine_num" : 10,
    "max_job_num" : [50,75,100],
    "lambda_rate" :[0.12,0.11,0.10],
    "seed":42,
    "fjsp_same_path":"experiment/fjsp/fjsp_same.json",
    "seed_list" : np.random.RandomState(42).randint(0, 1e6, size=1000000)
}
