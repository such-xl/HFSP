# callback.py
import os
from collections import defaultdict, Counter
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import csv


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, log_path="./episode_metrics.csv", verbose=0):
        super(EpisodeMetricsCallback, self).__init__(verbose)
        self.log_path = log_path
        self.header_written = False

    def _on_training_start(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps","lambda_rate","episode_reward", "tardiness"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "tardiness" in info and "episode_reward" in info:
                with open(self.log_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [self.num_timesteps,info["lambad_rate"],info["episode_reward"], sum(info["tardiness"])]
                    )
            # 记录到 TensorBoard
                # self.logger 是 BaseCallback 自动提供的 logger
                self.logger.record("rollout/episode_reward", info["episode_reward"])
                self.logger.record("rollout/tardiness", sum(info["tardiness"]))
                # --- 新增部分：记录 actions_count ---
                if info['actions_count'] is not None and isinstance(info['actions_count'], list):
                    for i, count in enumerate(info['actions_count']):
                        self.logger.record(f"actions_count/action_{i}", count)
                # --- 结束新增部分 ---
        return True

class ActionStatsOnlyCallback(BaseCallback):
    """
    仅用于动作统计的回调函数，可以与现有的EpisodeMetricsCallback配合使用
    """
    def __init__(self, action_space_size, verbose=0):
        super(ActionStatsOnlyCallback, self).__init__(verbose)
        self.action_space_size = action_space_size
        self.episode_actions = defaultdict(list)
        self.episode_lengths = defaultdict(int)
        
    def _on_step(self) -> bool:
        actions = self.locals.get('actions')
        dones = self.locals.get('dones', [False])
        
        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = actions.flatten()
            elif not isinstance(actions, (list, tuple)):
                actions = [actions]
                
            for env_id, action in enumerate(actions):
                self.episode_actions[env_id].append(int(action))
                self.episode_lengths[env_id] += 1
                
                if env_id < len(dones) and dones[env_id]:
                    self._log_action_stats(env_id)
                    
        return True
    
    def _log_action_stats(self, env_id):
        """记录动作统计到TensorBoard"""
        if len(self.episode_actions[env_id]) == 0:
            return
            
        actions = np.array(self.episode_actions[env_id])
        episode_length = len(actions)
        action_counts = Counter(actions)
        
        # 记录每个动作的统计
        for action in range(self.action_space_size):
            count = action_counts.get(action, 0)
            percentage = (count / episode_length) * 100 if episode_length > 0 else 0
            
            self.logger.record(f"actions/action_{action}_count", count)
            self.logger.record(f"actions/action_{action}_percentage", percentage)
        
        # 其他统计指标
        self.logger.record("actions/action_diversity", len(action_counts))
        if action_counts:
            most_common = max(action_counts, key=action_counts.get)
            self.logger.record("actions/most_common_action", most_common)
        
        # 清空记录
        self.episode_actions[env_id] = []
        self.episode_lengths[env_id] = 0
