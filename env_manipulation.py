
from envs.ReachCubeEnv import ReachCubeEnv

def do_env_sim():

    env = ReachCubeEnv(render=True)
    env.reset()

    max_step = 1000
    for step in range(max_step):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        #print("Observation:", observation)
        #print("Reward:", reward)

        if done or step % 100 == 0:
            print(f"Cube reached the target position at step: {step} with reward {reward}")
            env.reset()

        env.render()

if __name__ == '__main__':
    do_env_sim()