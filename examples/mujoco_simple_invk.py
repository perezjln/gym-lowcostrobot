import time, argparse

import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot.simulated_robot import SimulatedRobot


def do_simple_invk(robot_id="5dof"):

    if robot_id == "5dof":
        path_scene = "gym_lowcostrobot/assets/scene_one_cube.xml"
        joint_name = "joint5-pad"
        object_id = "red_box_joint"
        nb_dof = 5
        min_dist = 0.02
        max_dist = 0.12
    elif robot_id == "6dof":
        path_scene = "gym_lowcostrobot/assets/low_cost_robot/scene.xml"
        joint_name = "moving_side"
        object_id = "red_box_joint"
        nb_dof = 6
        min_dist = 0.02
        max_dist = 0.12
    else:
        path_scene = "gym_lowcostrobot/assets/scene_so_arm_6dof_one_cube.xml"
        joint_name = "Fixed_Gripper"
        object_id = "red_box_joint"
        nb_dof = 6
        min_dist = 0.12
        max_dist = 0.45


    m     = mujoco.MjModel.from_xml_path(path_scene)
    data  = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    with mujoco.viewer.launch_passive(m, data) as viewer:
        
        # Get the final position of the cube
        cube_pos = data.joint(object_id).qpos[:3]

        # Run the simulation
        while viewer.is_running():
            step_start = time.time()
            q_target_pos = robot.inverse_kinematics(ee_target_pos=cube_pos, joint_name=joint_name, nb_dof=nb_dof)
            robot.set_target_pos(q_target_pos)

            # Step the simulation forward
            mujoco.mj_step(m, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # Get the final position of the cube
            cube_pos = data.joint(object_id).qpos[:3]

            ee_id = m.body(joint_name).id
            ee_pos = data.geom_xpos[ee_id]

            print("Cube dist:", np.linalg.norm(cube_pos - ee_pos))
            if np.linalg.norm(cube_pos - ee_pos) < min_dist or np.linalg.norm(cube_pos - ee_pos) > max_dist:
                print("Cube reached the target position")
                data.joint(object_id).qpos[:3] = [np.random.rand() * 0.2, np.random.rand() * 0.2, 0.01]
                mujoco.mj_step(m, data)
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument("--robot", choices=["5dof", "6dof", "6dof_soarm"], default="5dof", help="Choose the lowcost robot type")
    args = parser.parse_args()

    do_simple_invk(args.robot)
