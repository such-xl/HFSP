import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# 读取reward数据
with open("record.json", "r") as f:
    record = json.load(f)
with open("record_sr.json", "r") as f:
    record_sr = json.load(f)
ur = record["utilization_rate"]
ur_sr = record_sr["utilization_rate"]

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

plt.xlabel("Episode")
plt.ylabel("Std Utilization")
plt.title("Std Agent Utilization")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Std_Utilization.png")
