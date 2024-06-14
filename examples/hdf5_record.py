import argparse

import tqdm

from gym_lowcostrobot.envs.reach_cube_env import ReachCubeEnv
from gym_lowcostrobot.envs.wrappers.record_hdf5 import RecordHDF5Wrapper


def do_record_hdf5(args):
    env = ReachCubeEnv(render_mode=None, action_mode="ee")
    env = RecordHDF5Wrapper(env, hdf5_folder=args.folder, length=1000, name_prefix="reach")
    env.reset()

    max_step = 20000
    for _ in tqdm.tqdm(range(max_step)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace video from HDF5 trace file")
    parser.add_argument("--folder", type=str, default="data/", help="Path to HDF5 folder")
    args = parser.parse_args()
    do_record_hdf5(args)
