import os
import random
job_file_root_path = os.path.dirname(os.path.abspath(__file__))+'/train_data'
files_and_dirs = os.listdir(job_file_root_path)
for i in range(100):
    print(random.choice(files_and_dirs))