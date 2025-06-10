import json
from params import PARAMS
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def print_filter(data):
    # kernel_size = 30
    # return np.convolve(data, np.ones(kernel_size)/kernel_size, mode='valid')
    return savgol_filter(data, window_length=33,polyorder=3)
    return data


# name_list = ['RL']
name_list = ["RL"]

# 为每个算法指定一个颜色
colors = {
    "RL": "tab:blue",
    # "SPT": "tab:orange",
    # "LPT": "tab:green",
    # "LRPT": "tab:red",
    # "FIFO": "tab:purple",
}


def read_json(name):
    if name == "RL":
        with open(
        #    f"HFSD/record/record_{20}_{50}_{40}_{name}.json",
            f"HFSD/records/record_{PARAMS['machine_num']}_{PARAMS['E_ave']}_{PARAMS['new_insert']}_{name}.json",
            "r",
        ) as f:
            return json.load(f)
    else:
        with open(f"HFSD/records/record_{name}.json", "r") as f:
            return json.load(f)


# 读取并平滑
record = {n: read_json(n) for n in name_list}
results = {
    name: {
        "Makespan": print_filter(record[name]["makespan"]),
        "slack_time": print_filter(record[name]["slack_time"]),
        "M_U": print_filter(record[name]["utiliaction"]),
        
    }
    for name in name_list
}

episodes = np.arange(1, len(record["RL"]["makespan"]) + 1)
metrics = ["Makespan", "slack_time", "M_U"]
titles = ["Makespan", "slack_time", "Machine Utilization"]

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
            linewidth=1.5,
        )
    ax.set_title(titles[i])
    ax.set_xlabel("Episode")
    ax.set_ylabel(titles[i])
    ax.legend(loc="best")  # 添加图例

plt.tight_layout()
plt.savefig("HFSD/plt/comparison_plot.png", dpi=300)
print("图像已保存到 HFSD/plt/comparison_plot.png")
