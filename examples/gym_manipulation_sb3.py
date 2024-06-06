import torch
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_lowcostrobot.envs.lift_cube_env import LiftCubeEnv
from gym_lowcostrobot.envs.reach_cube_env import ReachCubeEnv


def do_td3_reach():
    env = ReachCubeEnv()
    env = FilterObservation(env, ["arm_qpos", "object_qpos"])
    env = FlattenObservation(env)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="td3_reach_cube", progress_bar=True)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_ppo_reach(device="cpu", render=True):
    nb_parallel_env = 4
    envs = SubprocVecEnv(
        [
            lambda: FlattenObservation(FilterObservation(ReachCubeEnv(), ["arm_qpos", "object_qpos"]))
            for _ in range(nb_parallel_env)
        ]
    )

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1, device=device)
    model.learn(total_timesteps=int(1e3), tb_log_name="ppo_reach_cube", progress_bar=True)

    # Evaluate the agent
    env = FlattenObservation(FilterObservation(ReachCubeEnv(render_mode="human"), ["arm_qpos", "object_qpos"]))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=render)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_td3_lift():
    env = LiftCubeEnv()
    env = FilterObservation(env, ["arm_qpos", "object_qpos"])
    env = FlattenObservation(env)

    # Define the evaluation callback
    eval_env = LiftCubeEnv()
    eval_env = FilterObservation(eval_env, ["arm_qpos", "object_qpos"])
    eval_env = FlattenObservation(eval_env)

    eval_callback = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=10, deterministic=True, callback_on_new_best=None)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e3), tb_log_name="td3_lift_cube", callback=eval_callback, progress_bar=True)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_ppo_lift():
    nb_parallel_env = 4
    envs = SubprocVecEnv(
        [
            lambda: FlattenObservation(FilterObservation(LiftCubeEnv(), ["arm_qpos", "object_qpos"]))
            for _ in range(nb_parallel_env)
        ]
    )

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=int(1e3), tb_log_name="ppo_lift_cube", progress_bar=True)

    # Evaluate the agent
    env = FlattenObservation(FilterObservation(LiftCubeEnv(render_mode="human"), ["arm_qpos", "object_qpos"]))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    ## print available devices
    print("Available devices:")
    print(torch.cuda.device_count())

    do_td3_lift()
