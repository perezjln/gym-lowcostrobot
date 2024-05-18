from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_lowcostrobot.envs.lift_cube_env import LiftCubeEnv
from gym_lowcostrobot.envs.reach_cube_env import ReachCubeEnv


def do_td3_reach():
    do_render = False
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="td3_reach_cube")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_ppo_reach():
    nb_parallel_env = 4
    do_render = False
    envs = SubprocVecEnv(
        [lambda: ReachCubeEnv(render=do_render, max_episode_steps=200) for _ in range(nb_parallel_env)]
    )

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="ppo_reach_cube")

    # Evaluate the agent
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_td3_lift():
    do_render = False
    env = LiftCubeEnv(render=do_render, max_episode_steps=200, action_mode="ee")

    # Define the evaluation callback
    eval_env = LiftCubeEnv(render=do_render, max_episode_steps=200, action_mode="ee")
    eval_callback = EvalCallback(
        eval_env, eval_freq=1000, n_eval_episodes=10, deterministic=True, callback_on_new_best=None
    )

    # Define and train the TD3 agent
    model = TD3("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="td3_lift_cube", callback=eval_callback)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def do_ppo_lift():
    do_render = False
    nb_parallel_env = 8
    envs = SubprocVecEnv(
        [
            lambda: LiftCubeEnv(render=do_render, max_episode_steps=200, action_mode="ee")
            for _ in range(nb_parallel_env)
        ]
    )

    # Define and train the TD3 agent
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=int(1e5), tb_log_name="ppo_lift_cube")

    # Evaluate the agent
    env = ReachCubeEnv(render=do_render, max_episode_steps=200)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


if __name__ == "__main__":
    do_td3_lift()
