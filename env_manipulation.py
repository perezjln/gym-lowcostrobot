
from envs.ReachCubeEnv import ReachCubeEnv

def do_env_sim():

    env = ReachCubeEnv(render=True)
    env.reset()

    max_step = 200
    for step in range(max_step):
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)

        #print("Observation:", observation)
        #print("Reward:", reward)

        if done:
            if not truncated:
                print(f"Cube reached the target position at step: {step} with reward {reward}")
            else:
                print(f"Cube didn't reached the target position at step: {step} with reward {reward} but was truncated")
            env.reset()

        env.render()

if __name__ == '__main__':
    do_env_sim()