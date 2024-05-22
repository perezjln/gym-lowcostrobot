import time, argparse

import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot.simulated_robot import SimulatedRobot


def do_simple_invk_5dof():
    m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/scene_one_cube.xml")
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

            ee_id = m.body("joint5-pad").id
            ee_pos = data.geom_xpos[ee_id]

            print("Cube dist:", np.linalg.norm(cube_pos - ee_pos))
            if np.linalg.norm(cube_pos - ee_pos) < 0.06 or np.linalg.norm(cube_pos - ee_pos) > 0.12:
                print("Cube reached the target position")
                data.joint("red_box_joint").qpos[:3] = [np.random.rand() * 0.2, np.random.rand() * 0.2, 0.01]
                mujoco.mj_step(m, data)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def do_simple_invk_6dof():

    m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/scene_so_arm_6dof_one_cube.xml")
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:
        # Get the final position of the cube
        cube_pos = data.joint("red_box_joint").qpos[:3]

        # Run the simulation
        while viewer.is_running():
            step_start = time.time()
            q_target_pos = robot.inverse_kinematics(ee_target_pos=cube_pos, step=1.0, joint_name="Fixed_Gripper", nb_dof=6)
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

            ee_id = m.body("Moving_Jaw").id
            ee_pos = data.geom_xpos[ee_id]

            print("Cube dist:", np.linalg.norm(cube_pos - ee_pos))
            if np.linalg.norm(cube_pos - ee_pos) < 0.1 or np.linalg.norm(cube_pos - ee_pos) > 0.13:
                print("Cube reached the target position")

                data.qpos[:6] = 0.0
                data.joint("red_box_joint").qpos[:3] = [np.random.rand() * -0.2, np.random.rand() * -0.2, 0.01]
                mujoco.mj_forward(m, data)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument("--robot", choices=["5dof", "6dof"], default="5dof", help="Choose the lowcost robot type")
    args = parser.parse_args()

    if args.robot == "5dof":
        do_simple_invk_5dof()
    elif args.robot == "6dof":
        do_simple_invk_6dof()

