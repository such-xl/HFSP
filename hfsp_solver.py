import gymnasium as gym
from stable_baselines3 import PPO
from scheduling_env.hfsp import HFSPEnv
gym.envs.register(
    id="HFSP-v0",
    entry_point="scheduling_env.hfsp:HFSPEnv",
    # vector_entry_point="",
    max_episode_steps=400,
    reward_threshold=197.0,
)

# Parallel environments
env = gym.make("HFSP-v0")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")
