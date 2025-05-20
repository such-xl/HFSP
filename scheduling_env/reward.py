import numpy as np
import time
from collections import deque


class AsyncMachineUtilizationReward:
    def __init__(
        self,
        num_machines,
        w1=1,
        w2=0,
        safety_threshold=0.9,
        history_length=600,
        decay_factor=0.95,
    ):
        """
        初始化异步奖励函数计算器（无锁版本）

        参数:
        num_machines (int): 系统中的机器数量
        w1 (float): 平均机器利用率的权重
        w2 (float): 机器利用率标准差的权重
        safety_threshold (float): 安全负载阈值，超过这个值会被认为是过载
        history_length (int): 历史记录长度
        decay_factor (float): 时间衰减因子，用于权衡最近和较早的利用率数据
        """
        self.num_machines = num_machines
        self.w1 = w1
        self.w2 = w2
        self.safety_threshold = safety_threshold
        self.history_length = history_length
        self.decay_factor = decay_factor

        # 为每台机器初始化历史记录和时间戳
        self.machine_history = [
            {
                "utilization": deque(maxlen=history_length),
                "timestamp": deque(maxlen=history_length),
            }
            for _ in range(num_machines)
        ]

        # 系统级别的历史记录
        self.system_history = {
            "mean_utilization": deque(maxlen=history_length),
            "std_utilization": deque(maxlen=history_length),
            "timestamp": deque(maxlen=history_length),
        }

        # 最后一次系统更新的时间戳
        self.last_system_update = 0

    def normalize_reward(self, reward, scale=2.0):
        """
        使用sigmoid函数将奖励归一化到[-1,1]范围

        参数:
        reward (float): 原始奖励值
        scale (float): 控制sigmoid陡峭程度的缩放因子

        返回:
        float: 归一化后的奖励，范围在[-1,1]
        """
        # sigmoid函数返回(0,1)范围的值，乘以2再减1转换为(-1,1)
        normalized = 2 / (1 + np.exp(-scale * reward)) - 1
        return normalized

    def analyze_trend_change_rate(self, machine_id, current_time, window_size=5):
        """分析利用率趋势的变化率"""
        history = self.machine_history[machine_id]
        if len(history["utilization"]) < window_size:
            return 0.0

        # 获取最近的window_size个利用率
        recent_utils = list(history["utilization"])[-window_size:]
        recent_times = list(history["timestamp"])[-window_size:]

        # 计算相邻点之间的变化率
        change_rates = []
        for i in range(1, len(recent_utils)):
            time_diff = recent_times[i] - recent_times[i - 1]
            if time_diff > 0:  # 避免除以零
                util_change = recent_utils[i] - recent_utils[i - 1]
                rate = util_change / time_diff
                change_rates.append(rate)

        if not change_rates:
            return 0.0

        # 计算变化率的平均绝对值 - 越小表示越稳定
        avg_change_rate = np.mean(np.abs(change_rates))

        # 将变化率转换为奖励 - 变化率越小，奖励越高
        stability_reward = np.exp(-5 * avg_change_rate) - 0.5  # 范围约为[-0.5, 0.5]

        return stability_reward

    def update_machine_utilization(self, machine_id, utilization, timestamp=None):
        """
        更新单个机器的利用率

        参数:
        machine_id (int): 机器ID
        utilization (float): 当前利用率，范围[0,1]
        timestamp (float, 可选): 时间戳，默认为当前时间
        """
        if timestamp is None:
            raise ValueError("timestamp is None")

        self.machine_history[machine_id]["utilization"].append(utilization)
        self.machine_history[machine_id]["timestamp"].append(timestamp)

    def update_system_state(self, current_time, force=False):
        """
        更新系统状态（平均利用率和标准差）

        参数:
        force (bool): 是否强制更新，默认为False
        """

        # 如果距离上次更新时间过短且不强制更新，则跳过
        if (
            not force and current_time - self.last_system_update < 0.5
        ):  # 1秒的最小更新间隔
            return

        # 检查是否有足够的数据更新系统状态
        if all(
            len(self.machine_history[i]["utilization"]) > 0
            for i in range(self.num_machines)
        ):
            # 获取每台机器的最新利用率
            latest_utilization = np.array(
                [
                    self.machine_history[i]["utilization"][-1]
                    for i in range(self.num_machines)
                ]
            )

            # 计算系统平均利用率和标准差
            mean_utilization = np.mean(latest_utilization)
            std_utilization = np.std(latest_utilization)

            # 更新系统历史记录
            self.system_history["mean_utilization"].append(mean_utilization)
            self.system_history["std_utilization"].append(std_utilization)
            self.system_history["timestamp"].append(current_time)

            # 更新最后更新时间
            self.last_system_update = current_time

    def get_time_weighted_utilization(self, machine_id, current_time=None):
        """
        获取时间加权的机器利用率

        参数:
        machine_id (int): 机器ID
        current_time (float, 可选): 当前时间，默认为当前时间

        返回:
        float: 时间加权的利用率
        """
        if current_time is None:
            raise ("current_time is None")

        history = self.machine_history[machine_id]
        if not history["utilization"]:
            return 0.0

        # 计算时间权重
        time_diffs = np.array([current_time - t for t in history["timestamp"]])
        weights = np.exp(-time_diffs / 60.0)  # 10秒的时间尺度
        weights = weights / np.sum(weights)  # 归一化权重

        # 计算加权利用率
        weighted_utilization = np.sum(np.array(list(history["utilization"])) * weights)

        return weighted_utilization

    def calculate_machine_reward(self, machine_id, current_time=None):
        """
        计算单个机器的奖励

        参数:
        machine_id (int): 机器ID
        current_time (float, 可选): 当前时间，默认为当前时间

        返回:
        float: 机器的奖励
        dict: 详细奖励分解
        """
        if current_time is None:
            raise ("current_time is None")

        # 确保系统状态是最新的
        self.update_system_state(current_time)

        # 获取机器的时间加权利用率
        machine_utilization = self.get_time_weighted_utilization(
            machine_id, current_time
        )

        # 如果系统历史记录为空，则无法计算奖励
        if not self.system_history["mean_utilization"]:
            return 0.0, {"local_reward": 0.0, "global_reward": 0.0, "total_reward": 0.0}

        # 获取系统的平均利用率和标准差
        system_mean_utilization = self.system_history["mean_utilization"][-1]
        system_std_utilization = self.system_history["std_utilization"][-1]

        # 计算局部奖励（基于机器自身的利用率）
        local_reward = self.w1 * machine_utilization

        # 计算全局奖励（基于机器对系统平衡的贡献）
        # 利用率接近系统平均值，贡献较高
        balance_contribution = 1 - abs(
            machine_utilization - system_mean_utilization
        ) / max(system_mean_utilization, 0.01)
        global_reward = balance_contribution * (
            self.w1 * system_mean_utilization - self.w2 * system_std_utilization
        )

        # 总奖励
        total_reward = 0.1 * local_reward + 0.9 * global_reward - 0.5
        # 返回奖励和详细分解
        reward_details = {
            "local_reward": local_reward,
            "global_reward": global_reward,
            "total_reward": total_reward,
            "machine_utilization": machine_utilization,
            "system_mean_utilization": system_mean_utilization,
            "system_std_utilization": system_std_utilization,
        }

        return total_reward, reward_details

    def calculate_system_reward(self, current_time=None):
        """
        计算整个系统的奖励

        参数:
        current_time (float, 可选): 当前时间，默认为当前时间

        返回:
        float: 系统的总体奖励
        dict: 详细奖励分解
        """
        if current_time is None:
            raise ValueError("current_time is None")

        # 确保系统状态是最新的
        self.update_system_state(current_time, force=True)

        # 如果系统历史记录为空，则无法计算奖励
        if not self.system_history["mean_utilization"]:
            return 0.0, {"system_reward": 0.0}

        # 获取系统的平均利用率和标准差
        system_mean_utilization = self.system_history["mean_utilization"][-1]
        system_std_utilization = self.system_history["std_utilization"][-1]

        # 获取每台机器的最新利用率
        latest_utilization = np.array(
            [
                self.get_time_weighted_utilization(i, current_time)
                for i in range(self.num_machines)
            ]
        )

        max_utilization = np.max(latest_utilization)
        # 计算系统奖励
        system_reward = (
            self.w1 * system_mean_utilization - self.w2 * system_std_utilization
        )

        # 返回奖励和详细分解
        reward_details = {
            "system_mean_utilization": system_mean_utilization,
            "system_std_utilization": system_std_utilization,
            "max_utilization": max_utilization,
            "system_reward": system_reward,
        }

        return system_reward, reward_details


# 示例用法 - 模拟异步环境
def example_async_usage():
    # 初始化奖励函数计算器
    num_machines = 5
    reward_calculator = AsyncMachineUtilizationReward(num_machines)

    # 模拟异步更新
    for t in range(20):
        print(f"=== 时间步 {t+1} ===")

        # 模拟每台机器在不同时间点报告利用率
        for machine_id in range(num_machines):
            # 生成随机利用率
            if t < 10:
                # 前10步，模拟不平衡的利用率
                utilization = np.random.uniform(0.3, 0.9)
                if machine_id == 0:
                    utilization = np.random.uniform(0.85, 0.98)  # 第一台机器经常过载
                elif machine_id == 1:
                    utilization = np.random.uniform(0.1, 0.3)  # 第二台机器经常闲置
            else:
                # 后10步，模拟更平衡的利用率
                utilization = np.random.uniform(0.6, 0.8)

            # 模拟不同的更新时间
            current_time = time.time() + np.random.uniform(-0.5, 0.5)  # 模拟时间漂移

            # 更新机器利用率
            reward_calculator.update_machine_utilization(
                machine_id, utilization, current_time
            )

            # 计算并打印该机器的奖励
            reward, details = reward_calculator.calculate_machine_reward(machine_id)
            print(f"机器 {machine_id} 利用率: {utilization:.4f}, 奖励: {reward:.4f}")
            print(
                f"  局部奖励: {details['local_reward']:.4f}, 全局奖励: {details['global_reward']:.4f}"
            )
            print(f"  过载惩罚: {details['overload_penalty']:.4f}")

        # 每隔几个时间步计算系统总体奖励
        if t % 5 == 0 or t == 19:
            system_reward, system_details = reward_calculator.calculate_system_reward()
            print("\n系统总体统计:")
            print(f"平均利用率: {system_details['system_mean_utilization']:.4f}")
            print(f"利用率标准差: {system_details['system_std_utilization']:.4f}")
            print(f"最大利用率: {system_details['max_utilization']:.4f}")
            print(f"系统总体奖励: {system_reward:.4f}")

        print("-" * 50)


# 与强化学习框架集成的示例
def integration_with_rl_framework():
    """
    演示如何将奖励函数与强化学习框架集成
    """
    # 这里仅展示概念性代码，需要根据实际使用的RL框架调整

    # 初始化环境和奖励计算器
    num_machines = 10
    reward_calculator = AsyncMachineUtilizationReward(num_machines)

    # 假设的RL循环
    for episode in range(5):
        print(f"Episode {episode + 1}")

        # 重置环境
        # state = env.reset()

        for step in range(100):
            # 为每个机器做出决策
            for machine_id in range(num_machines):
                # 在实际应用中，这里应该使用您的RL智能体获取动作
                # action = agent.get_action(state, machine_id)

                # 模拟执行动作后的利用率变化
                utilization = np.random.uniform(
                    0.4, 0.9
                )  # 这应该是动作执行后的实际利用率

                # 更新利用率历史
                reward_calculator.update_machine_utilization(machine_id, utilization)

                # 计算奖励
                reward, details = reward_calculator.calculate_machine_reward(machine_id)

                # 在实际应用中，这里应该将奖励传递给RL智能体
                # agent.update(state, action, reward, next_state, machine_id)

                # 简单打印，在实际应用中可能需要记录日志
                if step % 20 == 0 and machine_id == 0:  # 只打印部分数据以避免输出过多
                    print(f"Step {step}, Machine {machine_id}, Reward: {reward:.4f}")

            # 每隔一定步数评估系统整体表现
            if step % 50 == 0:
                system_reward, details = reward_calculator.calculate_system_reward()
                print(
                    f"Step {step}, System Reward: {system_reward:.4f}, Mean Utilization: {details['system_mean_utilization']:.4f}"
                )

        print("-" * 50)


if __name__ == "__main__":
    # 运行异步示例
    example_async_usage()

    # 如果需要查看与RL框架集成的示例，取消下面的注释
    # integration_with_rl_framework()
