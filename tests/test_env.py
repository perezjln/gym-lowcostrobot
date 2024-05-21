import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_lowcostrobot  # noqa


@pytest.mark.parametrize(
    "env_id",
    [
        "LiftCube-v0",
        "PickPlaceCube-v0",
        "PushCube-v0",
        "ReachCube-v0",
        "Stack-v0",
    ],
)
def test_env_check(env_id):
    env = gym.make(env_id)
    check_env(env, skip_render_check=True)
    env.close()
