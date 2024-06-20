import argparse

import mujoco
import mujoco.viewer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    # You can easily load a dataset from a Hugging Face repository
    dataset = LeRobotDataset(args.repo_id)
    print(f"\n{dataset[0]['observation.images.cam_high'].shape=}")  # (4,c,h,w)
    print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
    print(f"{dataset[0]['action'].shape=}\n")  # (64,c)
    print(f"{len(dataset)=}\n")

    model = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/low_cost_robot_6dof/reach_cube.xml")
    data = mujoco.MjData(model)

    current_episode = 0
    print(f"Starting episode {current_episode}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run the simulation

        step = 0
        while viewer.is_running():
            if step in dataset.episode_data_index["from"]:
                current_episode += 1
                mujoco.mj_resetData(model, data)
                print(f"Starting episode {current_episode}")

            # Step the simulation forward
            data.ctrl = dataset[step]["observation.state"]
            mujoco.mj_step(model, data)

            viewer.sync()
            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trajectories from leRobotDataset hub")
    parser.add_argument("--repo_id", type=str, default="thomwolf/blue_sort", help="Path to HDF5 file")
    args = parser.parse_args()
    main()
