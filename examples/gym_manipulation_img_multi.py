import matplotlib.pyplot as plt

from gym_lowcostrobot.envs.push_cube_env import PushCubeEnv


def do_env_sim_image():
    env = PushCubeEnv(render_mode=None)
    env.reset()

    max_step = 1000
    for _ in range(max_step):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        for key in ["image_front", "image_top"]:
            print(key)
            plt.imshow(obs[key])
            plt.show()

        if terminated or truncated:
            env.reset()

        env.render()


if __name__ == "__main__":
    do_env_sim_image()
