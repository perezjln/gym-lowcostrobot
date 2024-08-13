import os, argparse
import numpy as np

import mujoco
import gymnasium as gym
import gym_lowcostrobot  # noqa


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


def real_to_mujoco(real_positions, inverted_joints=[], half_inverted_joints=[]):
    """
    Convert joint positions from real robot (in degrees) to Mujoco (in radians),
    with support for inverted joints.
    
    Parameters:
    real_positions (list or np.array): Joint positions in degrees.
    inverted_joints (list): List of indices for joints that are inverted.
    
    Returns:
    np.array: Joint positions in radians.
    """
    real_positions = np.array(real_positions)
    mujoco_positions = real_positions * (np.pi / 180.0)

    # Apply inversion if necessary
    for index in inverted_joints:
        mujoco_positions[index] += np.pi
        mujoco_positions[index] *= -1
    
    # Apply half inversion if necessary
    for index in half_inverted_joints:
        mujoco_positions[index] -= np.pi / 2.0
        mujoco_positions[index] *= -1
    
    return mujoco_positions


def do_read_pos(packetHandler, portHandler, ADDR_MX_PRESENT_POSITION):
    # Read present position
    real_pos = np.zeros(6)
    for current_id in range(6):
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, current_id + 1, ADDR_MX_PRESENT_POSITION)
        real_pos[current_id] = dxl_present_position*360/4096
    return real_to_mujoco(real_pos, inverted_joints=[1, 2, 3, 5], half_inverted_joints=[4])


def do_env_sim(args):

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

    env = gym.make("PickPlaceCube-v0", observation_mode="state", render_mode="human", action_mode="joint")
    env.reset()

    pos_arm = do_read_pos(packetHandler, portHandler, ADDR_MX_PRESENT_POSITION)
 
    # compute forward kinematics
    env.data.qpos[-6:] = pos_arm
    mujoco.mj_forward(env.model, env.data)
    env.viewer.sync()

    max_step = 1000000
    for _ in range(max_step):

        new_pos_arm = do_read_pos(packetHandler, portHandler, ADDR_MX_PRESENT_POSITION)
        action = (new_pos_arm - pos_arm)*100
        observation, reward, terminated, truncated, info = env.step(action)
        pos_arm = new_pos_arm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    parser.add_argument('--protocol_version', type=float, default=2.0, help='Protocol version (e.g., 1.0 or 2.0)')    
    args = parser.parse_args()

    do_env_sim(args)
