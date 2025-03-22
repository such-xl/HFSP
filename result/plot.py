import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

colors = [
    "#FF69B4",
    "#FFD700",
    "#FF4500",
    "#00FF7F",
    "#7FFF00",
    "#FF1493",
    "#00BFFF",
    "#FF8C00",
    "#FFB6C1",
    "#008080",
    "#800080",
    "#9932CC",
    "#FF6347",
    "#BA55D3",
    "#3CB371",
    "#a1d99b",
    "#FF00FF",
    "#a63603",
    "#228B22",
    "#6A5ACD",
    "#F0E68C",
    "#4682B4",
    "#E6E6FA",
    "#d62728",
]
color_index = 0
# 读取reward数据
with open("record_rl.json", "r") as f:
    record = json.load(f)
reward = record["reward"]
utilization_rate = record["utilization_rate"]
makespan = record["makespan"]

# 绘制折线图
plt.figure(figsize=(10, 6))
# for agent, values in reward.items():
#     if values:
#         values = savgol_filter(values, window_length=11, polyorder=2)
#         plt.plot(
#             values[:1000], label=agent, color=colors[color_index % len(colors)]
#         )  # 使用颜色列表中的颜色
#         color_index += 1

all_values = []
for values in reward.values():
    if values:
        all_values.append(values)
if all_values:
    mean_values = np.mean(all_values, axis=0)  # 直接计算均值
    # mean_values = np.tanh(mean_values)
    mean_values_move = savgol_filter(mean_values, window_length=11, polyorder=2)
    plt.plot(mean_values_move, label="Mean Reward", color="red", linestyle="--")

    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Mean Agent Reward")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("reward.png")
else:
    print("所有agent的reward数据都为空，无法绘制均值图。")
# 使用NumPy的corrcoef函数计算相关系数矩阵
correlation_matrix = np.corrcoef(makespan, all_values)
# 相关系数是矩阵对角线之外的元素
correlation = correlation_matrix[0, 1]
print("correlation:", correlation)
