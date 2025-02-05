import gymnasium as gym

env = gym.make('CarRacing-v2')
print("Observation space:", env.observation_space)

print("Action space:", env.action_space)

