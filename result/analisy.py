import numpy as np
import json
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def analisy(data_type, id):
    file_name = f"record_{data_type}_rl.json"
    with open(file_name, "r") as f:
        record_rl = json.load(f)
    makespan_rl = record_rl["makespan"]
    reward_rl = np.array(list(record_rl["reward"].values()))
    ur_rl = np.array(list(record_rl["utilization_rate"].values()))
    actor_loss = np.array(record_rl["actor_loss"])
    critic_loss = np.array(record_rl["critic_loss"])
    entropy = np.array(record_rl["entropy"])
    with open(f"record_{data_type}_sr.json", "r") as f:
        record_sr = json.load(f)
    makespan_sr = record_sr["makespan"]
    ur_sr = np.array(list(record_sr["utilization_rate"].values()))


    with open(f"record_{data_type}_system.json","r") as f:
        system_record = json.load(f)
    sys_reward = system_record["system_reward"]
    sys_tardiness = system_record["avg_tardiness"]

    sys_tardiness_rate = savgol_filter(system_record["tardiness_rate"], window_length=11, polyorder=2)
    tardiness = np.array(list(record_rl["tardiness"].values())[:100])

    col_sums = np.sum(tardiness, axis=0)
    col_counts = np.count_nonzero(tardiness, axis=0)
    result = np.zeros_like(col_sums,dtype=float)
    nonzero_cols = col_counts > 0
    result[nonzero_cols] = col_sums[nonzero_cols]/col_counts[nonzero_cols]

    fig2 = plt.figure(id + 1, figsize=(10, 8), dpi=600)
    plt.title("mean_reward")
    plt.plot(
        savgol_filter(np.mean(reward_rl, axis=0), window_length=110, polyorder=2),
        label="RL",
    )
    for k,reward in enumerate(reward_rl,1):
        plt.plot(savgol_filter(reward, window_length=91, polyorder=2), alpha=0.2, label = f"machine{k}")

    plt.savefig("fjsp_reward")

    fig = plt.figure(id, figsize=(10, 8), dpi=600)
    plt.subplot(321)
    plt.plot(sys_tardiness_rate)
    plt.legend()
    plt.title("sys_tardiness_rate")

    plt.subplot(322)
    plt.plot(savgol_filter(sys_tardiness,window_length=11,polyorder=2),label="sys_tardness")
    plt.plot(savgol_filter(result, window_length=11, polyorder=2), label="mean_tardiness")
    plt.legend()
    plt.title("sys_tardness")


    plt.subplot(323)
    plt.title("sys_reward")
    plt.plot(
        savgol_filter(sys_reward, window_length=110, polyorder=2),
        label="RL",
    )
    plt.legend()

    plt.subplot(324)
    plt.plot(
        savgol_filter(reward_rl.mean(axis=0), window_length=91, polyorder=2),
        label="mean_reward",
    )
    plt.title("mean_reward")
    plt.legend()

    plt.subplot(325)
    plt.plot(
        savgol_filter(np.sum(tardiness, axis=0), window_length=91, polyorder=2),
        label="sum tardiness",
    )
    plt.legend()
    plt.title("sum tardiness")

    plt.subplot(326)
    plt.title("train info")
    plt.plot(
        savgol_filter(actor_loss * 10, window_length=110, polyorder=2),
        label="actor_loss",
    )
    plt.plot(
        savgol_filter(critic_loss*10, window_length=91, polyorder=2), label="critic_loss"
    )
    plt.plot(
        savgol_filter(entropy * 10, window_length=91, polyorder=2), label="entropy"
    )
    plt.legend()

    correction, p_value = spearmanr(result,sys_reward)
    print(
        f"correlation between tardiness and sys_reward: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction,p_value = spearmanr(result, reward_rl.mean(axis=0))
    print(
        f"correlation between tardiness and G: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = spearmanr(sys_tardiness_rate, reward_rl.mean(axis=0))
    print(
        f"correlation betwee tardiness_rate and G: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = spearmanr(makespan_rl, ur_rl.mean(axis=0))
    print(
        f"correlation between makespan and ur.mean: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = pearsonr(makespan_rl, ur_rl.std(axis=0))
    print(
        f"correlation between makespan and ur.std: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = pearsonr(ur_rl.mean(axis=0),np.mean(reward_rl, axis=0))
    print(
        f"correlation between ur.mean and reward.mean: {correction:.4f}, p_value: {p_value:.2f}"
    )
    plt.tight_layout()
    plt.savefig(f"analisy_{data_type}")




if __name__ == "__main__":
    task_type = {
        # "jsp": "jsp.json",
        # "fjsp_diff": "fjsp_diff.json",
        "fjsp_same": "fjsp_same.json",
        # "cfjsp_diff": "cfjsp_diff.json",
        # "cfjsp_same": "cfjsp_same.json",
    }
    for i, k in enumerate(task_type.keys()):
        print(k)
        analisy(k, i)
