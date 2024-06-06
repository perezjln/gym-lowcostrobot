import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot.simulated_robot import SimulatedRobot


def displace_object(data, m, object_id, viewer, square_size=0.2, invert_y=False, origin_pos=[0, 0.1]):
    ### Sample a position in a square in front of the robot
    if not invert_y:
        x = np.random.uniform(origin_pos[0] - square_size / 2, origin_pos[0] + square_size / 2)
        y = np.random.uniform(origin_pos[1] - square_size / 2, origin_pos[1] + square_size / 2)
    else:
        x = np.random.uniform(origin_pos[0] + square_size / 2, origin_pos[0] - square_size / 2)
        y = np.random.uniform(origin_pos[1] + square_size / 2, origin_pos[1] - square_size / 2)

    # data.joint(object_id).qpos[:3] = [np.random.rand() * coef_sample + min_dist_obj, np.random.rand() * coef_sample + min_dist_obj, 0.01]
    data.joint(object_id).qpos[:3] = [x, y, 0.01]

    mujoco.mj_step(m, data)
    viewer.sync()


def do_simple_trajectory_end_effector(current_pos, target_pos):
    # Define the trajectory
    nb_points = 10
    traj = np.linspace(current_pos, target_pos, nb_points)
    return traj


def do_simple_invk(robot_id="6dof", do_reset=False):
    if robot_id == "6dof":
        path_scene = "gym_lowcostrobot/assets/low_cost_robot_6dof/scene_one_cube.xml"
        joint_name = "moving_side"
        object_id = "red_box_joint"
        nb_dof = 6
        min_dist = 0.02
        max_dist = 0.35
        invert_y = False
        square_size = 0.2
        origin_pos=[0, 0.2]
    else:
        return

    m = mujoco.MjModel.from_xml_path(path_scene)
    data = mujoco.MjData(m)
    robot = SimulatedRobot(m, data)

    m.opt.timestep = 1 / 10000

    with mujoco.viewer.launch_passive(m, data) as viewer:
        # Get the final position of the cube
        # displace_object(data, m, object_id, coef_sample, min_dist_obj, viewer)
        displace_object(data, m, object_id, viewer, invert_y=invert_y, square_size=square_size, origin_pos=origin_pos)
        cube_pos = data.joint(object_id).qpos[:3]

        # Run the simulation
        while viewer.is_running():
            step_start = time.time()
            q_target_pos = robot.inverse_kinematics_reg(ee_target_pos=cube_pos, joint_name=joint_name, nb_dof=nb_dof, step=0.2)
            q_target_pos[-1] = 0.0
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

            if do_reset:
                if np.linalg.norm(cube_pos - ee_pos) < min_dist or np.linalg.norm(cube_pos - ee_pos) > max_dist:
                    print("Cube reached the target position")

                    # displace_object(data, m, object_id, coef_sample, min_dist_obj, viewer)
                    mujoco.mj_resetData(m, data)
                    displace_object(
                        data, m, object_id, viewer, invert_y=invert_y, square_size=square_size, origin_pos=origin_pos
                    )

                    # Rudimentary time keeping, will drift relative to wall clock.
                    time_until_next_step = m.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument("--robot", choices=["6dof"], default="6dof", help="Choose the lowcost robot type")
    args = parser.parse_args()

    do_simple_invk(args.robot, do_reset=True)
