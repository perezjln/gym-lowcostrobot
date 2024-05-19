import gymnasium as gym
import gym_lowcostrobot

if __name__ == "__main__":

    for env_name in ["lowcostrobot-LiftCube-v0", "lowcostrobot-PickPlaceCube-v0", "lowcostrobot-PushCube-v0", "lowcostrobot-ReachCube-v0", "lowcostrobot-Stack-v0"]:
        env = gym.make(env_name)
        observation1, info = env.reset(seed=123)
        observation2, info = env.reset(seed=123)
        observation3, info = env.reset(seed=123)
        assert (observation1 == observation2).all()
        assert (observation2 == observation3).all()
        print(f"{env_name} is deterministic")
