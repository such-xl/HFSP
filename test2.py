import gymnasium as gym

dim_1 = 5
dim_2 = 5
space = gym.MultiDiscrete([dim_1, dim_2])
print(space)