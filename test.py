import os
import matplotlib.pyplot as plt
import json
import math
import numpy as np

filename = 'record2'
with open(f'logs/{filename}.json','r') as f:
    record = json.load(f)
with open('logs/record_sjf2.json','r') as f:
    sjf_record = json.load(f)['makespan']
data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/train_data/'
jobs_name = os.listdir(data_path)
for i,name in enumerate (jobs_name):
    sjf_makepan = sjf_record[name]
    makespan = [x for x in record['makespan'][name]]
    reward = [x for x in record['reward'][name]]
    if len(makespan) == 0:
        continue
    print(min(makespan))
    print(sjf_makepan)
    # 使用NumPy的corrcoef函数计算相关系数矩阵
    correlation_matrix = np.corrcoef(makespan, reward)
    # 相关系数是矩阵对角线之外的元素
    correlation = correlation_matrix[0, 1]
    plt.figure(i)
    plt.title(name+' '+filename + ' ' + str(correlation)+' '+str(min(makespan))+ ' '+str(sjf_makepan))
    plt.plot(range(len(makespan)),makespan,c='blue')
    plt.plot(range(len(makespan)),[sjf_makepan for _ in range(len(makespan))],c='red')
    print(makespan)
    plt.savefig(f'logs/imgs/{name}u.png')
    print(f"Pearson correlation coefficient: {name}", correlation)
    plt.figure(i+100)
    plt.plot(range(len(reward)),reward)
    plt.savefig(f'logs/imgs/{name}u_reward.png')