import numpy as np
import time
from collections import deque


class AsyncTardinessReward:
    def __init__(
        self,
        num_machines,
        w_local=0.4,
        w_global=0.6,
        w_tardiness_rate=0,
        w_tardiness_time=0.01,
        history_length=1000,
        decay_factor=0.95,
        urgency_factor=3.0,
    ):
        """
        初始化基于迟到指标的异步奖励函数计算器

        参数:
        num_machines (int): 系统中的机器数量
        w_local (float): 局部奖励的权重
        w_global (float): 全局奖励的权重
        w_tardiness_rate (float): 迟到率的权重
        w_tardiness_time (float): 迟到时间的权重
        history_length (int): 历史记录长度
        decay_factor (float): 时间衰减因子
        urgency_factor (float): 紧急度计算的因子，值越大对接近截止日期的作业越敏感
        """
        self.num_machines = num_machines
        self.w_local = w_local
        self.w_global = w_global
        self.w_tardiness_rate = w_tardiness_rate
        self.w_tardiness_time = w_tardiness_time
        self.history_length = history_length
        self.decay_factor = decay_factor
        self.urgency_factor = urgency_factor
        self.pre_tardiness = 0
        # 为每台机器初始化历史记录
        self.machine_history = [
            {
                "slack_changes": deque(maxlen=history_length),
                "jobs_processed": deque(maxlen=history_length),
                "timestamp": deque(maxlen=history_length),
            }
            for _ in range(num_machines)
        ]

        # 系统级别的历史记录
        self.system_history = {
            "total_jobs": 0,
            "tardy_jobs": 0,
            "total_tardiness": 0.0,
            "avg_slack": deque(maxlen=history_length),
            "tardiness_rate": deque(maxlen=history_length),
            "timestamp": deque(maxlen=history_length),
        }

        # 作业跟踪
        self.job_tracking = {}  # job_id -> {deadline, remaining_processing_time, etc.}

        # 最后一次系统更新的时间戳
        self.last_system_update = 0

    def normalize_reward(self, reward, scale=0.01):

        normalized = -np.log1p(reward)
        return normalized

    def update_job_info(self, job_id, deadline, remaining_time, current_time):
        """
        更新或添加作业信息

        参数:
        job_id: 作业ID
        deadline: 截止日期
        remaining_time: 剩余处理时间
        current_time: 当前时间
        """
        self.job_tracking[job_id] = {
            "deadline": deadline,
            "remaining_time": remaining_time,
            "last_update": current_time,
            "slack": deadline - current_time - remaining_time,
            "complete_time": -1,
        }

    def update_job_completion(self, job_id, completion_time, machine_id):
        """
        记录作业完成情况

        参数:
        job_id: 完成的作业ID
        completion_time: 完成时间
        machine_id: 完成该作业的机器ID

        返回:
        tuple: (是否迟到, 迟到时间)
        """
        if job_id not in self.job_tracking:
            return False, 0  # 如果没有跟踪该作业，返回默认值
        job_info = self.job_tracking[job_id]
        deadline = job_info["deadline"]

        # 计算是否迟到及迟到时间
        is_tardy = completion_time > deadline
        tardiness = max(0, completion_time - deadline)

        # 更新系统历史
        self.system_history["total_jobs"] += 1
        if is_tardy:
            self.system_history["tardy_jobs"] += 1
            self.system_history["total_tardiness"] += tardiness

        # 更新机器历史
        self.machine_history[machine_id]["jobs_processed"].append(
            {
                "job_id": job_id,
                "is_tardy": is_tardy,
                "tardiness": tardiness,
                "completion_time": completion_time,
            }
        )

        # 从跟踪中移除该作业
        self.job_tracking.pop(job_id, None)
        return is_tardy, tardiness, deadline

    def calculate_slack_change(
        self, machine_id, assigned_job_id, current_time, processing_time
    ):
        """
        计算因为机器决策导致的松弛时间变化

        参数:
        machine_id: 做决策的机器ID
        assigned_job_id: 被分配的作业ID
        current_time: 当前时间
        processing_time: 该作业在该机器上的处理时间

        返回:
        tuple: (被分配作业的松弛变化, 系统中其他作业的平均松弛变化)
        """
        if assigned_job_id not in self.job_tracking:
            return 0, 0

        # 保存决策前的系统状态
        pre_decision_slacks = {
            j_id: info["slack"] for j_id, info in self.job_tracking.items()
        }

        # 模拟决策执行
        assigned_job = self.job_tracking[assigned_job_id]
        old_slack = assigned_job["slack"]

        # 更新被分配作业的剩余时间和松弛时间
        assigned_job["remaining_time"] -= processing_time
        assigned_job["slack"] = (
            assigned_job["deadline"]
            - (current_time + processing_time)
            - assigned_job["remaining_time"]
        )

        # 更新其他作业的松弛时间（它们可能因为等待而减少）
        for job_id, job_info in self.job_tracking.items():
            if job_id != assigned_job_id:
                # 假设该作业可能需要等待processing_time
                job_info["slack"] = (
                    job_info["deadline"]
                    - (current_time + processing_time)
                    - job_info["remaining_time"]
                )

        # 计算松弛时间变化
        assigned_slack_change = assigned_job["slack"] - old_slack

        other_jobs = [j_id for j_id in self.job_tracking if j_id != assigned_job_id]
        if other_jobs:
            other_slack_changes = [
                self.job_tracking[j_id]["slack"] - pre_decision_slacks[j_id]
                for j_id in other_jobs
            ]
            avg_other_slack_change = np.mean(other_slack_changes)
        else:
            avg_other_slack_change = 0

        # 记录松弛变化
        self.machine_history[machine_id]["slack_changes"].append(
            {
                "assigned_job": assigned_job_id,
                "assigned_slack_change": assigned_slack_change,
                "avg_other_slack_change": avg_other_slack_change,
                "time": current_time,
            }
        )

        # 恢复系统状态（因为这只是模拟计算）
        for j_id, slack in pre_decision_slacks.items():
            if j_id in self.job_tracking:  # 检查作业是否仍在跟踪中
                self.job_tracking[j_id]["slack"] = slack

        return assigned_slack_change, avg_other_slack_change

    def update_system_state(self, current_time, force=False):
        """更新系统状态"""
        if not force and current_time - self.last_system_update < 1.0:
            return

        # 计算当前迟到率
        tardiness_rate = 0
        if self.system_history["total_jobs"] > 0:
            tardiness_rate = (
                self.system_history["tardy_jobs"] / self.system_history["total_jobs"]
            )

        # 计算系统中所有作业的平均松弛时间
        all_slacks = [job_info["slack"] for job_info in self.job_tracking.values()]
        avg_slack = np.mean(all_slacks) if all_slacks else 0

        # 更新系统历史
        self.system_history["tardiness_rate"].append(tardiness_rate)
        self.system_history["avg_slack"].append(avg_slack)
        self.system_history["timestamp"].append(current_time)

        self.last_system_update = current_time

    def calculate_urgency_factor(self, job_id):
        """计算作业的紧急度因子"""
        if job_id not in self.job_tracking:
            return 0

        slack = self.job_tracking[job_id]["slack"]
        # 负松弛时间表示作业很可能迟到
        if slack < 0:
            return 1.0  # 最高紧急度

        # 使用指数衰减函数计算紧急度
        urgency = np.exp(
            -self.urgency_factor
            * slack
            / max(1, self.job_tracking[job_id]["remaining_time"])
        )
        return urgency

    def calculate_machine_reward(
        self, machine_id, assigned_job_id, current_time, processing_time
    ):
        """
        计算单个机器的决策奖励

        参数:
        machine_id: 机器ID
        assigned_job_id: 被分配的作业ID
        current_time: 当前时间
        processing_time: 处理时间

        返回:
        float: 奖励值
        dict: 详细信息
        """
        # 确保系统状态是最新的
        self.update_system_state(current_time)

        # 计算松弛变化
        assigned_slack_change, avg_other_slack_change = self.calculate_slack_change(
            machine_id, assigned_job_id, current_time, processing_time
        )

        # 计算紧急度
        urgency = self.calculate_urgency_factor(assigned_job_id)

        # 局部奖励：基于被分配作业的松弛变化和紧急度
        # 紧急作业的松弛时间增加应该得到更高奖励
        local_reward = assigned_slack_change * (1 + urgency)

        # 全局奖励：基于系统中其他作业的平均松弛变化
        global_reward = avg_other_slack_change

        # 如果作业已完成，增加对迟到的惩罚
        completion_penalty = 0
        if (
            assigned_job_id in self.job_tracking
            and self.job_tracking[assigned_job_id]["remaining_time"] <= 0
        ):
            is_tardy, tardiness, deadline = self.update_job_completion(
                assigned_job_id, current_time, machine_id
            )
            if is_tardy:
                # 迟到惩罚：迟到率惩罚 + 迟到时间惩罚
                tardiness_rate_penalty = (
                    -self.w_tardiness_rate * 1.0
                )  # 每个迟到作业的惩罚
                tardiness_time_penalty = (
                    -self.w_tardiness_time * tardiness / max(1, deadline)
                )  # 归一化迟到时间
                completion_penalty = tardiness_rate_penalty + tardiness_time_penalty

        # 总奖励
        total_reward = self.w_local * local_reward + self.w_global * global_reward + 0
        # completion_penalty)

        # 归一化奖励
        normalized_reward = self.normalize_reward(total_reward)

        # 详细信息
        reward_details = {
            "assigned_job_id": assigned_job_id,
            "urgency": urgency,
            "assigned_slack_change": assigned_slack_change,
            "avg_other_slack_change": avg_other_slack_change,
            "local_reward": local_reward,
            "global_reward": global_reward,
            "completion_penalty": completion_penalty,
            "total_reward": total_reward,
            "normalized_reward": normalized_reward,
        }

        return normalized_reward, reward_details

    def calculate_system_reward(self, current_time):
        """
        计算整个系统的奖励

        参数:
        current_time: 当前时间

        返回:
        float: 系统奖励
        dict: 详细信息
        """
        # 强制更新系统状态
        self.update_system_state(current_time, force=True)

        # 计算当前迟到率和平均迟到时间
        tardiness_rate = 0
        tardiness_td = 0
        if self.system_history["total_jobs"] > 0:
            tardiness_rate = (
                self.system_history["tardy_jobs"] / self.system_history["total_jobs"]
            )
            if self.system_history["tardy_jobs"] > 0:
                total_tardiness = self.system_history["total_tardiness"]
                tardiness_td = total_tardiness - self.pre_tardiness
                self.pre_tardiness = total_tardiness

        # 计算系统中所有作业的平均松弛时间
        all_slacks = [job_info["slack"] for job_info in self.job_tracking.values()]
        avg_slack = np.mean(all_slacks) if all_slacks else 0

        # 统计负松弛时间作业比例（处于危险状态的作业）
        negative_slack_jobs = sum(1 for slack in all_slacks if slack < 0)
        negative_slack_rate = negative_slack_jobs / len(all_slacks) if all_slacks else 0

        # 系统奖励：基于迟到率、平均迟到时间和危险作业比例
        system_reward = (
            + self.w_tardiness_time * tardiness_td
            + 1 * negative_slack_rate
        )
        normalized_reward = self.normalize_reward(system_reward)

        # 详细信息
        reward_details = {
            "tardiness_rate": tardiness_rate,
            "total_tardiness": self.system_history["total_tardiness"],
            "avg_slack": avg_slack,
            "negative_slack_rate": negative_slack_rate,
            "system_reward": system_reward,
            "normalized_reward": normalized_reward,
        }

        return normalized_reward, reward_details

    def get_job_slack_urgency_info(self):
        """获取所有作业的松弛和紧急度信息，用于可视化和调试"""
        results = {}
        for job_id, info in self.job_tracking.items():
            urgency = self.calculate_urgency_factor(job_id)
            results[job_id] = {
                "slack": info["slack"],
                "urgency": urgency,
                "remaining_time": info["remaining_time"],
                "deadline": info["deadline"],
            }
        return results

    def get_machine_contribution(self, machine_id, window=10):
        """计算机器对系统性能的贡献度"""
        if len(self.machine_history[machine_id]["jobs_processed"]) < window:
            return 0.0

        # 获取最近处理的作业
        recent_jobs = list(self.machine_history[machine_id]["jobs_processed"])[-window:]

        # 计算非迟到作业比例
        non_tardy_rate = sum(1 for job in recent_jobs if not job["is_tardy"]) / len(
            recent_jobs
        )

        # 计算平均迟到时间
        avg_tardiness = (
            np.mean([job["tardiness"] for job in recent_jobs]) if recent_jobs else 0
        )

        # 计算贡献度：非迟到率高且迟到时间低的机器贡献度高
        contribution = non_tardy_rate * (1 - min(1, avg_tardiness / 100.0))
        return contribution


# 示例用法
# reward_calculator = AsyncTardinessReward(num_machines=5)
