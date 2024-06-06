import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.envs.base_env import BaseRobotEnv


class LiftCubeEnv(BaseRobotEnv):
    """
    ## Description

    The robot has to lift a cube with its end-effector. The episode is terminated when the cube is lifted above a
    threshold distance.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"object_qpos"`: the position of the cube, as (x, y, z)
    - `"object_qvel"`: the velocity of the cube, as (vx, vy, vz)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)

    Three observation modes are available: "state", "image" (default), and "both".

    | Key             | `"state"` | `"image"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"object_qpos"` | ✓         |           | ✓        |
    | `"object_qvel"` | ✓         |           | ✓        |
    | `"image_front"` |           | ✓         | ✓        |
    | `"image_top"`   |           | ✓         | ✓        |

    ## Reward

    The reward is the z position of the cube. The episode is terminated when the cube is lifted above a threshold
    distance.

    ## Reward

    The reward is the z position of the cube. The episode is terminated when the cube is lifted above a threshold
    distance.
    """

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None, obj_xy_range=0.15):
        super().__init__(
            xml_path=os.path.join(ASSETS_PATH, "scene_one_cube.xml"),
            observation_mode=observation_mode,
            action_mode=action_mode,
            render_mode=render_mode,
        )

        # Define the action space and observation space
        self.action_space = self.set_action_space_with_gripper()

        # Set the observations space
        observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "arm_qvel": spaces.Box(low=-10.0, high=10.0, shape=(6,)),
        }
        if observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
        if observation_mode in ["state", "both"]:
            observation_subspaces["object_qpos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
            observation_subspaces["object_qvel"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        self.threshold_height = 0.5
        self.set_object_range(obj_xy_range)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        self.data.qpos[:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Sample and set the object position
        object_pos = self.np_random.uniform(self.object_low, self.object_high).astype(np.float32)
        object_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qpos[6:13] = np.concatenate([object_pos, object_rot])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation_dict_one_object(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Get the new observation
        observation = self.get_observation_dict_one_object()

        # Get the height of the object
        object_z = self.data.qpos[8]

        # Compute the reward
        reward = object_z

        # Check if the target position is reached
        terminated = object_z > self.threshold_height

        return observation, reward, terminated, False, {}
