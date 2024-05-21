import gymnasium as gym

if __name__ == "__main__":
    for env_name in [
        "LiftCube-v0",
        "PickPlaceCube-v0",
        "PushCube-v0",
        "ReachCube-v0",
        "Stack-v0",
    ]:
        env = gym.make(env_name)
        observation1, info = env.reset(seed=123)
        observation2, info = env.reset(seed=123)
        observation3, info = env.reset(seed=123)
        assert (observation1 == observation2).all()
        assert (observation2 == observation3).all()
        print(f"{env_name} is deterministic")
