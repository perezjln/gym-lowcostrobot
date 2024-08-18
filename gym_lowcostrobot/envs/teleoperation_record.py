
import time
import numpy as np

import mujoco
import mujoco.viewer

import argparse
import copy
import os
import pathlib
import time

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub, save_meta_data
from tqdm import tqdm



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


class SimCamera:

    def __init__(self, model, data, id_camera, camera_index, fps=30, width=640, height=480):
        self.model = model
        self.data  = data
        self.camera_index = camera_index
        self.id_camera = id_camera
        self.is_connected = False
        self.fps = fps
        self.width = width
        self.height = height

        self.logs = {}
        self.logs["delta_timestamp_s"] = 1.0 / self.fps
        
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def async_read(self):
            self.renderer.update_scene(self.data, camera=self.id_camera)
            return self.renderer.render()


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
    def real_to_mujoco(real_positions, transforms, oppose):
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
stop = False

def key_callback(keycode):

    global current_motor_ids
    global stop

    print(f"Key pressed: {chr(keycode)}")
    #print(follower.data.qpos)

    if chr(keycode) in ["1", "2", "3", "4", "5", "6"]:
        current_motor_ids = int(chr(keycode))
        print(f"Current motor id: {current_motor_ids}")

    if chr(keycode) == "7":
        stop = True

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

    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--repo-id", type=str, default="jnm38")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the recording.")
    parser.add_argument(
        "--fps_tolerance",
        type=float,
        default=0.1,
        help="Tolerance in fps for the recording before dropping episodes.",
    )
    parser.add_argument(
        "--revision", type=str, default=CODEBASE_VERSION, help="Codebase version used to generate the dataset."
    )

    args = parser.parse_args()

    repo_id = args.repo_id
    num_episodes = args.num_episodes
    num_frames = args.num_frames
    revision = args.revision
    fps = args.fps
    fps_tolerance = args.fps_tolerance

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
    
    cameras = {
        "image_top":   SimCamera(id_camera="camera_top",   model=follower.model, data=follower.data, camera_index=0, fps=30, width=640, height=480),
        "image_front": SimCamera(id_camera="camera_front", model=follower.model, data=follower.data, camera_index=1, fps=30, width=640, height=480),
    }

    DATA_DIR = pathlib.Path("data_traces")
    out_data = DATA_DIR / repo_id

    # During data collection, frames are stored as png images in `images_dir`
    images_dir = out_data / "images"

    # After data collection, png images of each episode are encoded into a mp4 file stored in `videos_dir`
    videos_dir = out_data / "videos"
    meta_data_dir = out_data / "meta_data"

    # Create image and video directories
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    ep_fps = []
    id_from = 0
    id_to = 0

    obs_replay = {}
    obs_replay["observation"] = []
    obs_replay["action"] = []
    obs_replay["image_top"] = []
    obs_replay["image_front"] = []

    timestamps = []
    start_time = time.time()
    drop_episode = False

    ### start the teleoperation
    ep_idx = 0
    do_record_images = True
    with mujoco.viewer.launch_passive(follower.model, follower.data, key_callback=key_callback) as viewer:    

        robot = KochRobot(leader_arms   = {"main": leader},
                          follower_arms = {"main": follower},
                          cameras       = cameras,
                          calibration_path=args.calibration_path)
        robot.connect()
        
        while stop == False:

            obs_dict, action_dict = robot.teleop_step(record_data=True)
            obs_replay["observation"].append(copy.deepcopy(obs_dict["observation.state"]))
            obs_replay["action"].append(copy.deepcopy(action_dict["action"]))

            if do_record_images:
                obs_replay["image_top"].append(copy.deepcopy(obs_dict["observation.images.image_top"].numpy()))
                obs_replay["image_front"].append(copy.deepcopy(obs_dict["observation.images.image_front"].numpy()))

            timestamps.append(time.time() - start_time)

        num_frames = len(timestamps)

        # os.system('spd-say "stop"')
        if not drop_episode:
            # os.system(f'spd-say "saving episode"')
            ep_dict = {}

            # store images in png and create the video
            if do_record_images:
                for img_key in ["image_top", "image_front"]:
                    save_images_concurrently(
                        obs_replay[img_key],
                        images_dir / f"{img_key}_episode_{ep_idx:06d}",
                        args.num_workers,
                    )
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"

                    # store the reference to the video frame
                    ep_dict[f"observation.{img_key}"] = [{"path": f"videos/{fname}", "timestamp": tstp} for tstp in timestamps]

            state     = torch.tensor(np.array(obs_replay["observation"]))
            action    = torch.tensor(np.array(obs_replay["action"]))
            next_done = torch.zeros(num_frames, dtype=torch.bool)
            next_done[-1] = True

            ep_dict["observation.state"] = state
            ep_dict["action"] = action

            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
            ep_dict["frame_index"]   = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"]     = torch.tensor(timestamps)
            ep_dict["next.done"]     = next_done
            ep_dicts.append(ep_dict)
            ep_fps.append(num_frames / timestamps[-1])
            print(f"Episode {ep_idx} done, fps: {ep_fps[-1]:.2f}")

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames if args.keep_last else id_from + num_frames - 1)

            id_to = id_from + num_frames if args.keep_last else id_from + num_frames - 1
            id_from = id_to

            ep_idx += 1

    if do_record_images:
        # os.system('spd-say "encode video frames"')
        for ep_idx in range(num_episodes):
            for img_key in ["image_top", "image_front"]:
                encode_video_frames(
                    images_dir / f"{img_key}_episode_{ep_idx:06d}",
                    videos_dir / f"{img_key}_episode_{ep_idx:06d}.mp4",
                    ep_fps[ep_idx],
                )

    #os.system('spd-say "concatenate episodes"')
    data_dict = concatenate_episodes(ep_dicts)  # Since our fps varies we are sometimes off tolerance for the last frame

    features = {}

    if do_record_images:
        keys = [key for key in data_dict if "observation.image_" in key]
        for key in keys:
            features[key.replace("observation.image_", "observation.images.")] = VideoFrame()
            data_dict[key.replace("observation.image_", "observation.images.")] = data_dict[key]
            del data_dict[key]

    features["observation.state"] = Sequence(length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None))
    features["action"] = Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None))
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"]   = Value(dtype="int64", id=None)
    features["timestamp"]     = Value(dtype="float32", id=None)
    features["next.done"]     = Value(dtype="bool", id=None)
    features["index"]         = Value(dtype="int64", id=None)
    
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)

    info = {
        "fps": sum(ep_fps) / len(ep_fps),  # to have a good tolerance in data processing for the slowest video
        "video": 1,
    }
    
    #os.system('spd-say "from preloaded"')
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    #os.system('spd-say "compute stats"')
    stats = compute_stats(lerobot_dataset, num_workers=args.num_workers)

    #os.system('spd-say "save to disk"')
    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(out_data / "train"))

    args.push_to_hub = True
    if args.push_to_hub:
        hf_dataset.push_to_hub(repo_id, token=True, revision="main")
        hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision=revision)

        push_videos_to_hub(repo_id, videos_dir, revision="main")
        push_videos_to_hub(repo_id, videos_dir, revision=revision)
