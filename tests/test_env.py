import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_lowcostrobot  # noqa


@pytest.mark.parametrize("env_id", ["lowcostrobot-LiftCube-v0", "lowcostrobot-PickPlaceCube-v0", "lowcostrobot-PushCube-v0", "lowcostrobot-ReachCube-v0", "lowcostrobot-Stack-v0"])
def test_env_check(env_id):
    env = gym.make(env_id)
    check_env(env)
    env.close()
