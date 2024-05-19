import matplotlib.pyplot as plt

from gym_lowcostrobot.envs.lift_cube_env import LiftCubeEnv


def do_env_sim_image():
    env = LiftCubeEnv(render_mode=None, image_state="multi")
    env.reset()

    max_step = 1000
    for _ in range(max_step):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)

        for key, img in info["dict_imgs"].items():
            print(key)
            plt.imshow(img)
            plt.show()

        if terminated or truncated:
            env.reset()

        env.render()


if __name__ == "__main__":
    do_env_sim_image()
