import gymnasium as gym

env = gym.make("ReachCube-v0")

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()
env.close()
