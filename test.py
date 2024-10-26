import os
import matplotlib.pyplot as plt
import json
import math

with open('logs/record2iattn.json','r') as f:
    record = json.load(f)
data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/train_data/'
jobs_name = os.listdir(data_path)
for name in ['vla20.fjs']:
    makespan = [x for x in record['makespan'][name]]

    reward = [x for x in record['reward'][name]]
    if name == 'vla20.fjs':
        plt.title(name)
        plt.plot(range(len(makespan)),makespan,c='blue')
        print(makespan)
        # plt.plot(range(len(reward[0:500])),[x for x in reward[0:500]],c='r')
        plt.savefig('w.png')
        


    import numpy as np


    # 使用NumPy的corrcoef函数计算相关系数矩阵
    correlation_matrix = np.corrcoef(makespan, reward)

    # 相关系数是矩阵对角线之外的元素
    correlation = correlation_matrix[0, 1]

    print(f"Pearson correlation coefficient: {name}", correlation)