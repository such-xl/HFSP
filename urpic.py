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
plt.figure(figsize=(10, 6))
for v in ur.values():
    if len(v) > 0:
        values.append(v)
for v in ur_sr.values():
    if len(v) > 0:
        values_sr.append(v)
print(len(values))
print(len(values_sr))
values = np.array(values)
values_sr = np.array(values_sr)
print(values.shape)
print(values_sr.shape)
if True:
    mean_values = np.mean(values, axis=0)  # 直接计算均值
    mean_values = savgol_filter(mean_values, window_length=110, polyorder=2)
    mean_values_sr = np.mean(values_sr, axis=0)  # 直接计算均值
    mean_values_sr = savgol_filter(mean_values_sr, window_length=110, polyorder=2)
    plt.plot(mean_values, label="Mean Reward", color="red", linestyle="--")
    plt.plot(mean_values_sr, label="Mean Reward SR", color="blue", linestyle="--")

    plt.xlabel("Episode")
    plt.ylabel("Mean UR")
    plt.title("Mean Agent UR")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("UR2.png")
else:
    print("所有agent的reward数据都为空，无法绘制均值图。")
