
import time
import numpy as np
import mujoco


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
        path_scene="gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml"
    ):
        
        self.path_scene = path_scene
        self.model = mujoco.MjModel.from_xml_path(path_scene)
        self.data  = mujoco.MjData(self.model)
        self.is_connected = False
        self.motors = {}
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
            values.append(self.data.qpos[-6+idx-1:])

        if return_list:
            return values
        else:
            return values[0]


    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`.")
        values = []
        for name in motor_names:
            values.append(self.data.qpos[-6+self.motors[name][0]-1:])
        return values


    def _write_with_motor_ids(self, motor_models, motor_ids, data_name, values):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )        
        for idx, value in zip(motor_ids, values):
            self.data.qpos[-6+idx-1:] = value

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise SimRobotDeviceNotConnectedError(
                f"SimDynamixelMotorsBus({self.path_scene}) is not connected. You need to run `motors_bus.connect()`."
            )

        if motor_names is None:
            motor_names = self.motor_names
        
        if not isinstance(values, list):
            values = [values]

        for name, value in zip(motor_names, values):
            self.data.qpos[-6+self.motors[name][0]-1:] = value


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="Port for the leader motors")
    parser.add_argument("--calibration-path", type=str, default=None, help="Path to the robots calibration file")    
    args = parser.parse_args()

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
            )

    robot = KochRobot(leader_arms={"main": leader}, 
                    follower_arms={"main": follower},
                    calibration_path=args.calibration_path)

    robot.connect()
