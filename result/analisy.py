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

    # wt_rl = np.array(list(record_rl["wait_time"].values()))
    # wt_sr = np.array(list(record_sr["wait_time"].values()))

    fig = plt.figure(id, figsize=(10, 8), dpi=600)
    plt.subplot(321)
    plt.plot(savgol_filter(makespan_rl, window_length=91, polyorder=2), label="RL")
    plt.plot(savgol_filter(makespan_sr, window_length=91, polyorder=2), label="SR")
    plt.legend()
    plt.title("makespan")

    plt.subplot(322)
    # plt.plot(
    #     savgol_filter(wt_rl.mean(axis=0), window_length=91, polyorder=2), label="RL-wt"
    # )
    # plt.plot(
    #     savgol_filter(wt_sr.mean(axis=0), window_length=91, polyorder=2), label="SR-wt"
    # )
    # plt.legend()
    plt.title("wait time")


    plt.subplot(323)
    plt.title("reward")
    plt.plot(
        savgol_filter(np.mean(reward_rl, axis=0), window_length=110, polyorder=2),
        label="RL",
    )
    plt.legend()

    plt.subplot(324)
    plt.plot(
        savgol_filter(np.mean(ur_rl, axis=0), window_length=91, polyorder=2),
        label="RL-mean_ur",
    )
    plt.plot(
        savgol_filter(np.mean(ur_sr, axis=0), window_length=91, polyorder=2),
        label="SR-mean_ur",
    )
    plt.title("mean utilization rate")
    plt.legend()

    plt.subplot(325)
    plt.plot(
        savgol_filter(np.std(ur_rl, axis=0) * 5, window_length=91, polyorder=2),
        label="RL-std_ur",
    )
    plt.plot(
        savgol_filter(np.std(ur_sr, axis=0) * 5, window_length=191, polyorder=2),
        label="SR-std_ur",
    )

    plt.legend()
    plt.title("std utilization rate")

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

    correction, p_value = pearsonr(makespan_rl, ur_rl.max(axis=0))
    print(data_type)

    print(
        f"correlation between makespan and ur.max: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = spearmanr(makespan_rl, ur_rl.min(axis=0))
    print(
        f"correlation between makespan and ur.min: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = spearmanr(makespan_rl, ur_rl.mean(axis=0))
    print(
        f"correlation between makespan and ur.mean: {correction:.4f}, p_value: {p_value:.2f}"
    )
    correction, p_value = pearsonr(makespan_rl, ur_rl.std(axis=0))
    print(
        f"correlation between makespan and ur.std: {correction:.4f}, p_value: {p_value:.2f}"
    )


    # correction, p_value = pearsonr(ur_rl.max(axis=0), wt_rl.mean(axis=0))
    # print(
    #     f"correlation between ur.mean and wt.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(ur_rl.min(axis=0), wt_rl.mean(axis=0))
    # print(
    #     f"correlation between ur.max and wt.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(ur_rl.mean(axis=0), wt_rl.mean(axis=0))
    # print(
    #     f"correlation between ur.min and wt.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(ur_rl.std(axis=0), wt_rl.mean(axis=0))
    # print(
    #     f"correlation between ur.std and wt.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    correction, p_value = pearsonr(ur_rl.mean(axis=0),np.mean(reward_rl, axis=0))
    print(
        f"correlation between ur.mean and reward.mean: {correction:.4f}, p_value: {p_value:.2f}"
    )
    # print("====================================")
    # correction, p_value = pearsonr(makespan_rl,reward_rl.mean(axis=0))
    # print(
    #     f"correlation between makespan and reward.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(wt_rl.mean(axis=0), reward_rl.mean(axis=0))
    # print(
    #     f"correlation between wt.mean and reward.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(reward_rl.mean(axis=0), ur_rl.mean(axis=0))
    # print(
    #     f"correlation between reward.mean and ur.mean: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(reward_rl.mean(axis=0), ur_rl.max(axis=0))
    # print(
    #     f"correlation between reward.mean and ur.max: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(reward_rl.mean(axis=0), ur_rl.min(axis=0))
    # print(
    #     f"correlation between reward.mean and ur.min: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # correction, p_value = pearsonr(reward_rl.mean(axis=0), ur_rl.std(axis=0))
    # print(
    #     f"correlation between reward.mean and ur.std: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    # print("====================================")
    # fig.suptitle(
    #     f"{data_type} \n makespan and ur.max() correlation: {correction:.4f}, p_value: {p_value:.2f}"
    # )
    plt.tight_layout()
    plt.savefig(f"{data_type}.png")


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
