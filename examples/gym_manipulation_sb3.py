import gymnasium as gym
import torch
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_lowcostrobot  # noqa


def do_td3_push():
    env = gym.make("PushCube-v0", observation_mode="state", render_mode=None)
    env = FilterObservation(env, ["arm_qpos", "object_qpos", "target_qpos"])
    env = FlattenObservation(env)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="td3_push_cube", progress_bar=True)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def make_env():
    env = gym.make("PushCube-v0", observation_mode="state", render_mode=None)
    env = FilterObservation(env, ["arm_qpos", "cube_pos", "target_pos"])
    env = FlattenObservation(env)
    return env


def do_ppo_push(device="cpu", render=True):
    nb_parallel_env = 4
    envs = make_vec_env(make_env, n_envs=nb_parallel_env)

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1, device=device)
    model.learn(total_timesteps=int(1e5), tb_log_name="ppo_push_cube", progress_bar=True)

    # Evaluate the agent
    env_test = gym.make("PushCube-v0", observation_mode="state", render_mode=None)
    env_test = FilterObservation(env_test, ["arm_qpos", "cube_pos", "target_pos"])
    env_test = FlattenObservation(env_test)
    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=10, render=render)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    ## print available devices
    print("Available devices:")
    print(torch.cuda.device_count())

    do_ppo_push()
