import json
from params import PARAMS
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def print_filter(data):
    return savgol_filter(data, window_length=110, polyorder=2)

name_list = ['RL', 'SPT', 'LPT', 'LRPT', 'SRPT']

# 为每个算法指定一个颜色
colors = {
    'RL':    'tab:blue',
    'SPT':    'tab:orange',
    'LPT':    'tab:green',
    'LRPT':   'tab:red',
    'SRPT':   'tab:purple',
}

def read_json(name):
    if name =="RL":
        with open(f"HFSD/record/record_{PARAMS['machine_num']}_{PARAMS['E_ave']}_{PARAMS['new_insert']}_{name}.json", "r") as f:
            return json.load(f)
    else:
        with open(f"HFSD/record/record_{name}.json", "r") as f:
            return json.load(f)

# 读取并平滑
record = {n: read_json(n) for n in name_list}
results = {
    name: {
        "Makespan":         print_filter(record[name]["makespan"]),
        "idle_time_ratio":  print_filter(record[name]["idle_time_ratio"]),
        "M_U":              print_filter(record[name]["utiliaction_rates"]),
        "tradiness":        print_filter(record[name]["tradiness"]),
    }
    for name in name_list
}

episodes = np.arange(1, len(record['RL']["makespan"]) + 1)
metrics = ["Makespan", "idle_time_ratio", "M_U", "tradiness"]
titles  = ["Makespan", "Idle Time Ratio", "Machine Utilization", "Tardiness"]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    ax = axs[i]
    for name in name_list:
        ax.plot(
            episodes,
            results[name][metric],
            label=name,
            color=colors[name],
            linewidth=1.5
        )
    ax.set_title(titles[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel(titles[i])
    ax.legend(loc="best")   # 添加图例

plt.tight_layout()
plt.savefig("HFSD/plt/comparison_plot.png", dpi=300)
print("图像已保存到 HFSD/plt/comparison_plot.png")
