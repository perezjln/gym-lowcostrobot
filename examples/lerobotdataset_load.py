import argparse, time
import torch

import mujoco
import mujoco.viewer
from gym_lowcostrobot.simulated_robot import SimulatedRobot

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():

    # You can easily load a dataset from a Hugging Face repository
    dataset = LeRobotDataset(args.repo_id)
    print(f"\n{dataset[0]['observation.image'].shape=}")  # (4,c,h,w)
    print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
    print(f"{dataset[0]['action'].shape=}\n")  # (64,c)

    # Finally, our datasets are fully compatible with PyTorch dataloaders 
    # and samplers because they are just PyTorch datasets.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )
    for batch in dataloader:
        print(f"{batch['observation.image'].shape=}")  # (32,4,c,h,w)
        print(f"{batch['observation.state'].shape=}")  # (32,8,c)
        print(f"{batch['action'].shape=}")  # (32,64,c)
        break

    m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/scene_so_arm_6dof_one_cube.xml")
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:
        # Run the simulation
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # Step the simulation forward
            robot.set_target_qpos(group_qpos[step][0:6])
            mujoco.mj_step(m, data)

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step += 1
            print(group_qpos[step][0:6])
            # step = step % len(group_qpos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trajectories from leRobotDataset hub")
    parser.add_argument("--repo_id", type=str, default="thomwolf/blue_sort", help="Path to HDF5 file")
    args = parser.parse_args()
    main()
