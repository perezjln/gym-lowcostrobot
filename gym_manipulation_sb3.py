
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from envs.tasks.ReachCubeEnv import ReachCubeEnv

def do_td3():

    do_render = False
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="td3_reach_cube")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_ppo():

    nb_parallel_env = 4
    do_render = False
    envs = SubprocVecEnv([lambda: ReachCubeEnv(render=do_render, max_episode_steps=200) for _ in range(nb_parallel_env)])

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="ppo_reach_cube")

    # Evaluate the agent
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == '__main__':
    do_td3()