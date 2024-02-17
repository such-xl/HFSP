import gymnasium as gym
import torch
env = gym.make('CartPole-v1',render_mode='human')
print(env.action_space)
observation, info = env.reset(seed=42)
steps = 0
for _ in range(0):
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        print("Episode finished after {} steps".format(steps))
        observation, info = env.reset()
        steps = 0
    else:
        steps += 1
    env.render()   
env.close()
a = torch.tensor([[0.7,0.8]])
print(a)
b=a.max(1).values
print(b)