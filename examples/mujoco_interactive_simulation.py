import argparse
import time

import mujoco
import mujoco.viewer


def do_interactive_sim_5dof():
    m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/scene_one_cube.xml")
    data = mujoco.MjData(m)

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

def do_interactive_sim_6dof():
    m = mujoco.MjModel.from_xml_path("gym_lowcostrobot/assets/scene_so_arm_6dof_one_cube.xml")
    data = mujoco.MjData(m)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument("--robot", choices=["5dof", "6dof"], default="5dof", help="Choose the lowcost robot type")
    args = parser.parse_args()

    if args.robot == "5dof":
        do_interactive_sim_5dof()
    elif args.robot == "6dof":
        do_interactive_sim_6dof()