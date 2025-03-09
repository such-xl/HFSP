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
env: HFSPEnv = gym.make("HFSP-v0")

model = PPO.load("ppo_cartpole")

obs, _ = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, terminated, info = env.step(action)
    dones = dones or terminated
    # env.render()
print(env.current_time)
