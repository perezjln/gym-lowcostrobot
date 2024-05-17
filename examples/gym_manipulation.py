from gym_lowcostrobot.envs.reach_cube_env import ReachCubeEnv


def do_env_sim():
    do_render = True
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)
    env.reset()

    max_step = 1000000
    for _ in range(max_step):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

        # print("Observation:", observation)
        # print("Reward:", reward)

        if do_render:
            env.render()

        if done:
            if not truncated:
                print(f"Cube reached the target position at step: {env.current_step} with reward {reward}")
            else:
                print(
                    f"Cube didn't reached the target position at step: {env.current_step} with reward {reward} but was truncated"
                )
            env.reset()


if __name__ == "__main__":
    do_env_sim()
