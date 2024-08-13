"""
Dynamixel SDK Example Script

This script interacts with Dynamixel motors using the Dynamixel SDK. It reads the present position of the motors and displays it on the console.

Command-Line Arguments:
- `--device`: The port name where the Dynamixel motor is connected (default: '/dev/ttyACM0').
- `--protocol_version`: The protocol version used by the Dynamixel motor (default: 2.0). Typically, this would be 1.0 or 2.0 depending on your hardware.

Usage Example:
python dynamixel_leader_read.py --device /dev/ttyUSB0 --protocol_version 2.0

Dependencies:
- dynamixel_sdk: Ensure you have the Dynamixel SDK installed. You can usually install it via pip or by following the instructions from the Dynamixel SDK documentation.

Note:
- This script supports both Windows and Unix-based systems. The `getch` function has been adapted to handle different platforms.
- The default settings are provided for convenience, but you can customize them according to your setup.

Ensure that the Dynamixel motor is properly connected to the specified port and that the settings match your hardware configuration.
One can use Dynamixel Wizzard 2 to configure the firmware of the motors : https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/

Example inspired from: https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/
"""


### Real: All strait position of the motors: 181.58203125 266.572265625 91.93359375 181.318359375 90.0 167.6953125
### Real: Left position of the motors:       9.033203125 180.0 176.8359375 92.021484375 177.1875 167.6953125

### Mujoco: All strait position of the motors: -0.03142 -1.56422 1.5423 -0.0044892 -0.000443796 0.0523634' (6 last on the qpos)
### Mujoco: Left position of the motors:       -1.571 -0.0137084 -0.00423796 1.53604 -1.18804e-06 0.052353' (6 last on the qpos)

import argparse
import os

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

def main(args):
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

    while True:
        print("Press any key to continue! (or press ESC to quit!)")
        if getch() == chr(0x1b):
            break

        while True:
            # Read present position
            lst_pos = []
            for current_id in range(6):
                dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, current_id + 1, ADDR_MX_PRESENT_POSITION)
                lst_pos.append(str(dxl_present_position*360/4096))

            print(' '.join(lst_pos))

    # Close port
    portHandler.closePort()

if __name__ == "__main__":
    # Set up argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Dynamixel SDK Example')
    parser.add_argument('--device', type=str, default='/dev/ttyACM0', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    parser.add_argument('--protocol_version', type=float, default=2.0, help='Protocol version (e.g., 1.0 or 2.0)')

    args = parser.parse_args()
    main(args)
