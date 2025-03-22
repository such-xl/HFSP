import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# 读取reward数据
with open("record_rl.json", "r") as f:
    record = json.load(f)
with open("record_sr.json", "r") as f:
    record_sr = json.load(f)
ur = record["utilization_rate"]
ur_sr = record_sr["utilization_rate"]
makespan = record["makespan"]
makespan_sr = record_sr["makespan"]
actor_loss = record["actor_loss"]
critic_loss = record["critic_loss"]
entropy = record["entropy"]
values = []
values_sr = []
for v in ur.values():
    if len(v) > 0:
        values.append(v)
for v in ur_sr.values():
    if len(v) > 0:
        values_sr.append(v)

mean_ur = np.mean(values, axis=0)
std_ur = np.std(values, axis=0)
mean_ur_sr = np.mean(values_sr, axis=0)
std_ur_sr = np.std(values_sr, axis=0)
mean_ur = savgol_filter(mean_ur, window_length=110, polyorder=2)
mean_ur_sr = savgol_filter(mean_ur_sr, window_length=110, polyorder=2)
std_ur = savgol_filter(std_ur, window_length=110, polyorder=2)
std_ur_sr = savgol_filter(std_ur_sr, window_length=110, polyorder=2)
makespan_sr = savgol_filter(makespan_sr, window_length=11, polyorder=2)
makespan = savgol_filter(makespan, window_length= 110, polyorder=2)
plt.figure(figsize=(10, 6))
plt.plot(mean_ur, label="Mean UR by RL", color="blue", linestyle=":")
plt.plot(mean_ur_sr, label="Mean UR by SR", color="red", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Utilization")
plt.title("Mean Agent Utilization")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Mean_Utilization.png")

plt.figure(figsize=(10, 6))
plt.plot(std_ur, label="Mean UR by RL", color="blue", linestyle=":")
plt.plot(std_ur_sr, label="Mean UR by SR", color="red", linestyle="--")
plt.axhline(y=np.mean(std_ur), color="blue", linewidth=0.1)

plt.axhline(y=np.mean(std_ur_sr), color="red", linewidth=0.1)
print(np.mean(std_ur))
print(np.mean(std_ur_sr))
plt.ylabel("Std Utilization")
plt.title("Std Agent Utilization")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Std_Utilization.png")

plt.figure(figsize=(10, 6))
plt.plot(makespan, label="makespan by RL", color="blue", linestyle=":")
plt.plot(makespan_sr, label="makespan by SR", color="red", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Makespan")
plt.title("Makespan")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Makespan.png")

actor_loss = savgol_filter(actor_loss, window_length=11, polyorder=2)
critic_loss = savgol_filter(critic_loss, window_length=11, polyorder=2)
entropy = savgol_filter(entropy, window_length=11, polyorder=2)

plt.figure(figsize=(10, 6))

# plt.plot(actor_loss, label="actor_loss", color="blue", linestyle="--")
plt.plot(critic_loss, label="critic_loss", color="red", linestyle="--")
# plt.plot(entropy, label="entropy", color="green", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Loss or entrop")
plt.title("train info")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Train_info.png")

makespan = record["makespan"]
# 使用NumPy的corrcoef函数计算相关系数矩阵
correlation_matrix = np.corrcoef(makespan,np.max(values, axis=0))
# 相关系数是矩阵对角线之外的元素
correlation = correlation_matrix[0, 1]
print("correlation:", correlation)

