from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id="ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    kwargs={"reward_type": "reward_type", "control_type": "control_type"},
    max_episode_steps=200,
)
