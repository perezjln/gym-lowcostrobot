import h5py
import time

import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot.simulated_robot import SimulatedRobot

def do_replay_hdf5():

    # Specify the path to your HDF5 file
    file_path = 'data/episode_46.hdf5'
    with h5py.File(file_path, 'r') as file:

        m = mujoco.MjModel.from_xml_path("assets/scene_one_cube.xml")
        data = mujoco.MjData(m)
        robot = SimulatedRobot(m, data)

        group_action = file['action']
        group_qpos = file['observations/qpos']

        with mujoco.viewer.launch_passive(m, data) as viewer:

            # Run the simulation
            step = 0
            while viewer.is_running():
                step_start = time.time()

                # Step the simulation forward
                robot.set_target_pos(group_action[step][0:5])
                mujoco.mj_step(m, data)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                step += 1
                step = step % len(group_qpos)


if __name__ == "__main__":
    do_replay_hdf5()
