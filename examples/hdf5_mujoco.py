import argparse
import time

import h5py
import mujoco
import mujoco.viewer

from gym_lowcostrobot.simulated_robot import SimulatedRobot


def do_replay_hdf5(args):
    # Specify the path to your HDF5 file
    with h5py.File(args.file_path, "r") as file:
        m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/low_cost_robot_6dof/reach_cube.xml")
        data = mujoco.MjData(m)
        robot = SimulatedRobot(m, data)

        # group_action = file["action"]
        group_qpos = file["observations/qpos"]

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
    parser = argparse.ArgumentParser(description="Trace video from HDF5 trace file")
    parser.add_argument("--file_path", type=str, default="data/episode_49.hdf5", help="Path to HDF5 file")
    args = parser.parse_args()
    do_replay_hdf5(args)
