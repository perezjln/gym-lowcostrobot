import matplotlib.pyplot as plt
from gym_lowcostrobot.envs.lift_cube_env import LiftCubeEnv


def do_env_sim_image():
    env = LiftCubeEnv(render=False, image_state=True)
    env.reset()

    max_step = 1000
    for _ in range(max_step):
        action = env.action_space.sample()
        _, _, done, _, info = env.step(action)

        for key, img in info["dict_imgs"].items():
            print(key)
            plt.imshow(img)
            plt.show()

        if done:
            env.reset()

        env.render()


if __name__ == "__main__":
    do_env_sim_image()
