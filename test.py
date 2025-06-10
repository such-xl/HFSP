import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
import numpy as np

# 假设你的文件路径如下
df = pd.read_csv("./ppo_logs/main_new3.csv")

rewards = df['episode_reward']
timesteps = df['timesteps']
tardiness = df['tardiness']
lambda_rate = df['lambda_rate']
# 假设你已经读取了CSV文件到df中
# 获取唯一的lambda_rate值
unique_lambda_rates = df['lambda_rate'].unique()

# 设置图形的大小和布局（可选）
plt.figure(figsize=(12, 8))

# 为每个lambda_rate绘制reward的图
for i,lambda_rate in enumerate(unique_lambda_rates,1):
    j = i*2 - 1
    # 筛选当前lambda_rate的数据
    filtered_df = df[df['lambda_rate'] == lambda_rate]
    reward = filtered_df['episode_reward'].to_numpy()
    tards = filtered_df['tardiness'].to_numpy()
    # 绘制reward
    plt.subplot(3,2,j)  # 计算子图位置
    plt.plot(reward, label=f'reward_lambda{lambda_rate}')
    plt.plot(savgol_filter(reward,window_length=51,polyorder=3))
    plt.title(f'Reward for lambda_rate={lambda_rate}')
    plt.xlabel('Episode Index')  # 假设你有一个episode的索引，否则可以使用其他合适的x轴标签

    plt.subplot(3,2,j+1)
    plt.plot(tards,label = f'tardiness_lambed{lambda_rate}' )
    plt.plot(savgol_filter(tards, window_length=51, polyorder=3))
    plt.title(f'Tardiness for lambda_rate={lambda_rate}')
    plt.xlabel('Episode Index')

# 调整子图之间的间距（可选）
plt.tight_layout()

plt.savefig("tardiness_reward.png")

correction, p_value = pearsonr(tardiness, rewards)
print(
    f"correlation pearsonr: {correction:.4f}, p_value: {p_value:.2f}"
)
correction, p_value = spearmanr(tardiness, rewards)
print(
    f"correlation spearmanr: {correction:.4f}, p_value: {p_value:.2f}"
)