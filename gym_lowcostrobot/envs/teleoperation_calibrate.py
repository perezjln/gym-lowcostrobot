
import argparse
import numpy as np

import mujoco
import mujoco.viewer

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.koch import KochRobot


## Define the simulated robot

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
    
    """
    The SimDynamixelMotorsBus class allows to efficiently read and write to the attached motors simulated in mujoco scene. 
    
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
    def real_to_mujoco(real_positions, transforms, oppose):
        """
        Convert real positions to mujoco positions

        Parameters:
        - real_positions: list of joint positions in degrees
        - transforms: list of joint transforms in radians
        - oppose: list of joint oppositions (1 or -1)
        """
        real_positions = np.array(real_positions)
        mujoco_positions = real_positions * (np.pi / 180.0)

        for id in range(6):
            mujoco_positions[id] = transforms[id] + mujoco_positions[id]
            mujoco_positions[id] *= oppose[id]

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



## test the leader motors reading
def test_read_leader_position():

    # Define the leader motors
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
    
    # connect the leader motors
    leader.connect()

    # read the motors position
    while True:
        print(leader.read("Present_Position", 
                          ["shoulder_pan", "shoulder_lift", 
                           "elbow_flex", "wrist_flex", 
                           "wrist_roll", "gripper"]))

    # disconnect the leader motors
    leader.disconnect()


## Mujoco keyboard callback
# [1-6] to select the current controlled motor using the keyboard
# [8] to increase the current motor position
# [9] to decrease the current motor position

current_motor_ids=1
def key_callback(keycode):
    global current_motor_ids

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

    PATH_SCENE = "gym_lowcostrobot/assets/low_cost_robot_6dof/pick_place_cube.xml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0", help="Port for the leader motors")
    parser.add_argument("--calibration-path", type=str, default=".cache/calibration/koch.pkl", help="Path to the robots calibration file")  
    parser.add_argument("--test-leader", action="store_true", help="Test the leader motors")
    args = parser.parse_args()

    ## test the leader motors reading 
    if args.test_leader:
        test_read_leader_position()
        exit()

    ## Instantiate the leader arm
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

    ## Instantiate the follower arm which is simulated
    follower = SimDynamixelMotorsBus(
                path_scene=PATH_SCENE,
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

    # Start the mujoco viewer
    with mujoco.viewer.launch_passive(follower.model, follower.data, key_callback=key_callback) as viewer:    

        # Define the robot as usual, with the leader and follower arms
        # One need to put the robot instantiation inside the mujoco viewer context
        # becasue the viewer is needed for the calibration that occurs in the robot instantiation
        robot = KochRobot(leader_arms={"main": leader},
                            follower_arms={"main": follower},
                            calibration_path=args.calibration_path)

        # Connect the robot
        robot.connect()

        # Test the teleoperation
        while True:
            robot.teleop_step()
