
import numpy as np
import gymnasium as gym
import gym_lowcostrobot

def do_env_sim():
    env = gym.make("PickPlaceCube-v0", observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    max_step = 1000000
    for step in range(max_step):

        action = np.asarray([0.0, 0.0, 0.0, 0.0])
        observation, reward, terminated, truncated, info = env.step(action)

        # print("Observation:", observation)
        # print("Reward:", reward)

        env.render()
        if terminated:
            if not truncated:
                print(f"Cube reached the target position at step: {env.current_step} with reward {reward}")
            else:
                print(
                    f"Cube didn't reached the target position at step: {env.current_step} with reward {reward} but was truncated"
                )
            env.reset()


if __name__ == "__main__":
    do_env_sim()
