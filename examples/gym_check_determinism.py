import gymnasium
from gymnasium.wrappers.filter_observation import FilterObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation

import gym_lowcostrobot  # noqa

if __name__ == "__main__":
    for env_name in [
        "LiftCube-v0",
        "PickPlaceCube-v0",
        "PushCube-v0",
        "ReachCube-v0",
        "Stack-v0",
    ]:
        env = gymnasium.make(env_name)
        env = FilterObservation(env, ["arm_qpos", "object_qpos"])
        env = FlattenObservation(env)

        observation1, info = env.reset(seed=123)
        observation2, info = env.reset(seed=123)
        observation3, info = env.reset(seed=123)
        assert (observation1 == observation2).all()
        assert (observation2 == observation3).all()
        print(f"{env_name} is deterministic")
