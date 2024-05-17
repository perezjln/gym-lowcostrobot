import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env



@pytest.mark.parametrize("env_id", ["LowCostRobot-v0", "LowCostRobot-v1"])
def test_env_check(env_id):
    env = gym.make(env_id)
    check_env(env)
    env.close()