import time
import numpy as np

import mujoco
import mujoco.viewer

from envs.interface import SimulatedRobot

def do_simple_invk():

    m = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:

        # Get the final position of the cube
        cube_id = m.body("box").id
        cube_pos = data.geom_xpos[cube_id]

        # Run the simulation
        while viewer.is_running():

            step_start = time.time()

            q_target_pos = robot.inverse_kinematics(ee_target_pos=cube_pos, joint_name="joint5-pad")
            robot.set_target_pos(q_target_pos)

            # Step the simulation forward
            mujoco.mj_step(m, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # Get the final position of the cube
            cube_id = mujoco.mj_name2id(m, 1, "box")
            cube_pos = data.xpos[cube_id]

            ee_id = mujoco.mj_name2id(m, 1, "joint5-pad")
            ee_pos = data.xpos[ee_id]

            print("Cube dist:", np.linalg.norm(cube_pos - ee_pos))
            if np.linalg.norm(cube_pos - ee_pos) < 0.26:
                print("Cube reached the target position")
                cube_pos = np.random.rand(3) * 2 - 1
                data.xpos[cube_id] = cube_pos

                mujoco.mj_step(m, data)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == '__main__':
    do_simple_invk()