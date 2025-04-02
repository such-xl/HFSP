# class A:
#     def __init__(self):
#         self.arr = [1, 2]

#     def pust(self, x):
#         self.arr.append(x)


# agent = A()
# agents = [agent] * 10

# for a in agents:
#     a.pust(3)
#     print(a.arr)

import numpy as np
num_jobs = 10
job_features = np.random.rand(num_jobs, 128)
machine_features = np.random.rand(5, 15)
print(len(job_features[0]))
print(job_features)
print(job_features.shape)
print(len(machine_features[0]))
print(machine_features)
print(machine_features.shape)