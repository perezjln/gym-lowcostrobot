import gymnasium as gym

import gym_lowcostrobot  # noqa

env = gym.make("LiftCube-v0", render_mode=None, observation_mode="state", action_mode="ee")

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    if terminated or truncated:
        env.reset()
env.close()
