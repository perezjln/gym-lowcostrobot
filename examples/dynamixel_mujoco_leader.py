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
        path_scene = "gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml"
    else:
        return

    m = mujoco.MjModel.from_xml_path(path_scene)
    #m.opt.timestep = 1 / 10000
    data = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, data) as viewer:

        ## position object in front of the robot
        data.joint("cube").qpos[:3] = [0.0, -0.1, 0.01]
        mujoco.mj_step(m, data)
        viewer.sync()

        # Run the simulation
        while viewer.is_running():

            step_start = time.time()

            # Read present position
            real_pos = np.zeros(6)
            for current_id in range(6):
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, current_id + 1, ADDR_MX_PRESENT_POSITION)
                real_pos[current_id] = dxl_present_position*360/4096

            # check limits
            # self.check_joint_limits(self.data.qpos)

            # compute forward kinematics
            data.qpos[-6:] = real_to_mujoco(real_pos, inverted_joints=[1, 2, 3, 5], half_inverted_joints=[4])
            mujoco.mj_step(m, data)
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

    do_sim(args.robot)