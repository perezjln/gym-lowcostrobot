import gymnasium as gym

import gym_lowcostrobot  # noqa


def do_env_sim():
    env = gym.make("PushCube-v0", observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    max_step = 1000000
    for _ in range(max_step):
        action = env.action_space.sample()
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
