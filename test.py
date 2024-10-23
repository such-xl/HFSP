import matplotlib.pyplot as plt
import json
import math
with open('logs/record2attn.json','r') as f:
    record = json.load(f)
makespan = [x for x in record['makespan']['rla15.fjs']]

reward = [x for x in record['reward']['rla15.fjs']]
plt.plot(range(len(makespan[0:500])),makespan[0:500],c='blue')
plt.plot(range(len(reward[0:500])),[x for x in reward[0:500]],c='r')
rewardt = [math.log(x) for x in makespan]
plt.savefig('w.png')


import numpy as np


# 使用NumPy的corrcoef函数计算相关系数矩阵
correlation_matrix = np.corrcoef(makespan, reward)

# 相关系数是矩阵对角线之外的元素
correlation = correlation_matrix[0, 1]

print("Pearson correlation coefficient:", correlation)