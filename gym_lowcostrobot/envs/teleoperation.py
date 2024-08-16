
import time
import numpy as np

import mujoco
import mujoco.viewer


class SimRobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class SimRobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class SimDynamixelMotorsBus:
    
    # TODO(rcadene): Add a script to find the motor indices without DynamixelWizzard2
    """
    The DynamixelMotorsBus class allows to efficiently read and write to the attached motors. It relies on
    the python dynamixel sdk to communicate with the motors. For more info, see the [Dynamixel SDK Documentation](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

    A DynamixelMotorsBus instance requires a port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/common/robot_devices/motors/dynamixel.py
    >>> Finding all available ports for the DynamixelMotorsBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> Remove the usb cable from your DynamixelMotorsBus and press Enter when done.
    >>> The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "xl330-m288"

    motors_bus = DynamixelMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        motors,
        path_scene="gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml"
    ):
        
        self.path_scene = path_scene
        self.model = mujoco.MjModel.from_xml_path(path_scene)
        self.data  = mujoco.MjData(self.model)
        self.is_connected = False
        self.motors = motors
        self.logs = {}

    def connect(self):
        self.is_connected = True

    def reconnect(self):
        self.is_connected = True

    def are_motors_configured(self):
        return True

    def configure_motors(self):
        print("Configuration is done!")

    def find_motor_indices(self, possible_ids=None):
        return [1, 2, 3, 4, 5, 6]

    def set_bus_baudrate(self, baudrate):
        return

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, dynamixel xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        # Convert from unsigned int32 original range [0, 2**32[ to centered signed int32 range [-2**31, 2**31[
        values = values.astype(np.int32)
        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        return values

    def _read_with_motor_ids(self, motor_models, motor_ids, data_name):
        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        values = []
        for idx in motor_ids:
            values.append(self.data.qpos[-6+idx-1])

        if return_list:
            return values
        else:
            return values[0]


    def read(self, data_name, motor_names: str | list[str] | None = None):

        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`.")

        values = []

        if motor_names is None:
            for idx in range(1, 7):
                values.append(self.data.qpos[idx-6-1])
        else:
            for name in motor_names:
                idx_motor = self.motors[name][0]-6-1
                values.append(self.data.qpos[idx_motor])

        return np.asarray(values)


    def _write_with_motor_ids(self, motor_models, motor_ids, data_name, values):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )        
        for idx, value in zip(motor_ids, values):
            self.data.qpos[idx-6-1] = value

    @staticmethod
    def real_to_mujoco(real_positions, transforms, oppose, inverted_joints=[], half_inverted_joints=[], oppose_joint=[], positive_half_inverted_joints=[]):
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

        for id in range(6):
            mujoco_positions[id] = transforms[id] + mujoco_positions[id]
            mujoco_positions[id] *= oppose[id]


        """
        # Apply inversion if necessary
        for index in inverted_joints:
            mujoco_positions[index] += np.pi + np.pi / 2.0
            mujoco_positions[index] *= -1
        
        # Apply half inversion if necessary
        for index in half_inverted_joints:
            mujoco_positions[index] -= np.pi / 2.0
            #mujoco_positions[index] *= -1

        # Apply half inversion if necessary
        for index in positive_half_inverted_joints:
            mujoco_positions[index] = np.pi / 2.0
            #mujoco_positions[index] *= -1

        # Apply half inversion if necessary
        for index in oppose_joint:
            mujoco_positions[index] *= -1
        """

        return mujoco_positions


    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):

        #print(data_name, values, motor_names)

        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )
        
        if data_name in ["Torque_Enable", "Operating_Mode", "Homing_Offset", "Drive_Mode", "Position_P_Gain", "Position_I_Gain", "Position_D_Gain"]:
            return

        if motor_names is None or len(motor_names) == 6:
            self.data.qpos[-6:] = self.real_to_mujoco(values, transforms=[0, 
                                                                          -np.pi / 2.0,
                                                                          np.pi + np.pi / 2.0,
                                                                          0,
                                                                          np.pi -np.pi / 2.0,
                                                                          0], 
                                                                          oppose=[-1,1,-1,1,-1,-1])

            mujoco.mj_step(follower.model, follower.data)
            viewer.sync()


    def disconnect(self):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. Try running `motors_bus.connect()` first."
            )

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()



import argparse
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.koch import KochRobot

def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

def test_teleoperate(robot: KochRobot, fps: int | None = None, teleop_time_s: float | None = None):

    if not robot.is_connected:
        robot.connect()

    with mujoco.viewer.launch_passive(robot.follower_arms["main"].model, robot.follower_arms["main"].data) as viewer:

        ## position object in front of the robot
        robot.follower_arms["main"].data.joint("cube").qpos[:3] = [0.0, -0.1, 0.01]
        mujoco.mj_step(robot.follower_arms["main"].model, 
                       robot.follower_arms["main"].data)
        viewer.sync()

        start_teleop_t = time.perf_counter()

        # Run the simulation
        while viewer.is_running():

            start_loop_t = time.perf_counter()
            robot.teleop_step()

            if fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t

            if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
                break

def test_read_leader_position():
    leader = DynamixelMotorsBus(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl330-m077"),
                    "shoulder_lift": (2, "xl330-m077"),
                    "elbow_flex": (3, "xl330-m077"),
                    "wrist_flex": (4, "xl330-m077"),
                    "wrist_roll": (5, "xl330-m077"),
                    "gripper": (6, "xl330-m077"),
                },
            )

    leader.connect()
    while True:
        print(leader.read("Present_Position", 
                          ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                           "wrist_flex", "wrist_roll", "gripper"]))

    leader.disconnect()



current_motor_ids=1
def key_callback(keycode):
    global current_motor_ids

    #print(f"Key pressed: {chr(keycode)}")
    #print(follower.data.qpos)

    if chr(keycode) in ["1", "2", "3", "4", "5", "6"]:
        current_motor_ids = int(chr(keycode))
        print(f"Current motor id: {current_motor_ids}")

    if chr(keycode) == "8":
        idx_motor = current_motor_ids-6-1
        follower.data.qpos[idx_motor] += 0.1
        mujoco.mj_forward(follower.model, 
                follower.data)
        viewer.sync()

    if chr(keycode) == "9":
        idx_motor = current_motor_ids-6-1
        follower.data.qpos[idx_motor] -= 0.1
        mujoco.mj_forward(follower.model, 
                follower.data)
        viewer.sync()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="Port for the leader motors")
    parser.add_argument("--calibration-path", type=str, default=".cache/calibration/koch.pkl", help="Path to the robots calibration file")  
    parser.add_argument("--test-leader", action="store_true", help="Test the leader motors")
    args = parser.parse_args()

    ## test the leader motors reading 
    if args.test_leader:
        test_read_leader_position()
        exit()

    ## test the teleoperation
    leader = DynamixelMotorsBus(
                port=args.leader_port,
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl330-m077"),
                    "shoulder_lift": (2, "xl330-m077"),
                    "elbow_flex": (3, "xl330-m077"),
                    "wrist_flex": (4, "xl330-m077"),
                    "wrist_roll": (5, "xl330-m077"),
                    "gripper": (6, "xl330-m077"),
                },
            )

    follower = SimDynamixelMotorsBus(
                path_scene="gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml",
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl430-w250"),
                    "shoulder_lift": (2, "xl430-w250"),
                    "elbow_flex": (3, "xl330-m288"),
                    "wrist_flex": (4, "xl330-m288"),
                    "wrist_roll": (5, "xl330-m288"),
                    "gripper": (6, "xl330-m288"),
                },
            )

    #with mujoco.viewer.launch(follower.model, follower.data) as viewer:
    with mujoco.viewer.launch_passive(follower.model, follower.data, key_callback=key_callback) as viewer:    

        robot = KochRobot(leader_arms={"main": leader}, 
                        follower_arms={"main": follower},
                        calibration_path=args.calibration_path)

        robot.connect()
        
        while True:
            robot.teleop_step()