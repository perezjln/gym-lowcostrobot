
import matplotlib.pyplot as plt
from gym_lowcostrobot.tasks.reach_cube_env import ReachCubeEnv

def do_env_sim_image():

    env = ReachCubeEnv(render=False, image_state=True)
    env.reset()

    max_step = 1000
    for _ in range(max_step):
        action = env.action_space.sample()
        _, _, done, _, info = env.step(action)

        plt.imshow(info["img"]) 
        plt.show()

        if done:
            env.reset()

        env.render()

if __name__ == '__main__':
    do_env_sim_image()
