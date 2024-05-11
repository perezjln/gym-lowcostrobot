import time
import numpy as np

import mujoco
import mujoco.viewer

from envs.SimulatedRobot import SimulatedRobot

def do_simple_invk():

    m = mujoco.MjModel.from_xml_path('low_cost_robot/scene_one_cube.xml')
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:

        # Get the final position of the cube
        cube_pos = data.joint("red_box_joint").qpos[:3]

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
            cube_pos = data.joint("red_box_joint").qpos[:3]

            ee_id    = m.body("joint5-pad").id
            ee_pos   = data.geom_xpos[ee_id]

            print("Cube dist:", np.linalg.norm(cube_pos - ee_pos))
            if np.linalg.norm(cube_pos - ee_pos) < 0.06 or np.linalg.norm(cube_pos - ee_pos) > 0.12:

                print("Cube reached the target position")
                data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]
                mujoco.mj_step(m, data)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == '__main__':
    do_simple_invk()