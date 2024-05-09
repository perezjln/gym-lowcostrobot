import time
import numpy as np

import mujoco
import mujoco.viewer

from envs.interface import SimulatedRobot

def do_interactive_sim():

    m = mujoco.MjModel.from_xml_path('low_cost_robot/scene_one_cube.xml')
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:

        # Run the simulation
        while viewer.is_running():

            step_start = time.time()

            # Step the simulation forward
            mujoco.mj_step(m, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    do_interactive_sim()