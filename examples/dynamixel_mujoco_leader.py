import os, argparse, time

import mujoco
import mujoco.viewer
import numpy as np


# Function to get a single character input
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from dynamixel_sdk import *  # Uses Dynamixel SDK library






import gymnasium as gym
import gym_lowcostrobot  # noqa

def do_env_sim():

    # Control table address
    ADDR_MX_PRESENT_POSITION = 132

    # Protocol version
    PROTOCOL_VERSION = args.protocol_version  # Use the protocol version from command line

    # Configuration from command line arguments
    DEVICENAME = args.device

    # Initialize PortHandler instance
    portHandler = PortHandler(DEVICENAME)

    # Initialize PacketHandler instance
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()


    env = gym.make("PushCube-v0", observation_mode="state", render_mode="human", action_mode="joint")
    env.reset()

    max_step = 1000000
    for _ in range(max_step):

        # Read present position
        action = np.zeros(6)
        for current_id in range(6):
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, current_id + 1, ADDR_MX_PRESENT_POSITION)
            action[current_id] = np.uint32(dxl_present_position)

        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        # print("Observation:", observation)
        # print("Reward:", reward)

        env.render()
        if terminated:
            if not truncated:
                print(f"Cube reached the target position at step: {env.current_step} with reward {reward}")
            else:
                print(
                    f"Cube didn't reached the target position at step: {env.current_step} with reward {reward} but was truncated"
                )
            env.reset()

    # Close port
    portHandler.closePort()


def do_sim(robot_id="6dof"):

    # Control table address
    ADDR_MX_PRESENT_POSITION = 132

    # Protocol version
    PROTOCOL_VERSION = args.protocol_version  # Use the protocol version from command line

    # Configuration from command line arguments
    DEVICENAME = args.device

    # Initialize PortHandler instance
    portHandler = PortHandler(DEVICENAME)

    # Initialize PacketHandler instance
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()

    if robot_id == "6dof":
        path_scene = "gym_lowcostrobot/assets/low_cost_robot_6dof/reach_cube.xml"
    else:
        return

    m = mujoco.MjModel.from_xml_path(path_scene)
    m.opt.timestep = 1 / 10000
    data = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, data) as viewer:

        # Run the simulation
        while viewer.is_running():

            step_start = time.time()

            #if getch() == chr(0x1b):
            #    break

            # Read present position
            for current_id in range(6):
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, current_id + 1, ADDR_MX_PRESENT_POSITION)
                data.qpos[current_id] = dxl_present_position

            print(data.qpos)

            # check limits
            # self.check_joint_limits(self.data.qpos)

            # compute forward kinematics
            mujoco.mj_forward(m, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Close port
    portHandler.closePort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument("--robot", choices=["6dof"], default="6dof", help="Choose the lowcost robot type")
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    parser.add_argument('--protocol_version', type=float, default=2.0, help='Protocol version (e.g., 1.0 or 2.0)')    
    args = parser.parse_args()

    #do_sim(args.robot)
    do_env_sim()
