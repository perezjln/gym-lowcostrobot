
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from envs.ReachCubeEnv import ReachCubeEnv

def do_env_sim():

    do_render = False
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5))

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    do_env_sim()