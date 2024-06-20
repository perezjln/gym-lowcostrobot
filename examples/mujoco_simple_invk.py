import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot.simulated_robot import LevenbegMarquardtIK


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

    mujoco.mj_forward(m, data)
    viewer.sync()


def do_simple_invk(robot_id="6dof", do_reset=False):
    if robot_id == "6dof":
        path_scene = "gym_lowcostrobot/assets/low_cost_robot_6dof/reach_cube.xml"
        joint_name = "moving_side"
        object_id = "cube"
        min_dist = 0.065
        max_dist = 0.35
        invert_y = False
        square_size = 0.2
        origin_pos = [0, 0.2]
    else:
        return

    m = mujoco.MjModel.from_xml_path(path_scene)
    data = mujoco.MjData(m)

    m.opt.timestep = 1 / 10000

    with mujoco.viewer.launch_passive(m, data) as viewer:
        # Get the final position of the cube
        # displace_object(data, m, object_id, coef_sample, min_dist_obj, viewer)
        displace_object(data, m, object_id, viewer, invert_y=invert_y, square_size=square_size, origin_pos=origin_pos)
        cube_pos = data.joint(object_id).qpos[:3]

        # Run the simulation
        while viewer.is_running():
            step_start = time.time()

            g_ik = LevenbegMarquardtIK(m, data, tol=min_dist)
            g_ik.calculate(body_id=m.body(joint_name).id, goal=cube_pos, viewer=viewer)

            # Get the final position of the cube
            error = np.linalg.norm(np.subtract(cube_pos, data.body(m.body(joint_name).id).xpos))

            if do_reset:
                if error < min_dist or error > max_dist:
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
