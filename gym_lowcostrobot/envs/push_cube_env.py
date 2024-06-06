import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.envs.base_env import BaseRobotEnv


class PushCubeEnv(BaseRobotEnv):
    """
    ## Description

    The robot has to push a cube to a target position. The episode is terminated when the cube is within a threshold
    distance from the target position.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 5-dimensional box

    | Index | Action                      | Type (unit) | Min  | Max |
    | ----- | --------------------------- | ----------- | ---- | --- |
    | 0     | Joint 1 (base to shoulder)  | Float (rad) | -1.0 | 1.0 |
    | 1     | Joint 2 (shoulder to elbow) | Float (rad) | -1.0 | 1.0 |
    | 2     | Joint 3 (elbow to wrist)    | Float (rad) | -1.0 | 1.0 |
    | 3     | Joint 4 (wrist to gripper)  | Float (rad) | -1.0 | 1.0 |
    | 4     | Joint 5 (gripper)           | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 3-dimensional box

    | Index | Action | Type (unit) | Min  | Max |
    | ----- | ------ | ----------- | ---- | --- |
    | 0     | X      | Float (m)   | -1.0 | 1.0 |
    | 1     | Y      | Float (m)   | -1.0 | 1.0 |
    | 2     | Z      | Float (m)   | -1.0 | 1.0 |

    ## Observation space

    | Index | Observation                              | Type (unit) | Min   | Max  |
    | ----- | ---------------------------------------- | ----------- | ----- | ---- |
    | 0     | Angle of 1st joint 1 (base to shoulder)  | Float (rad) | -3.14 | 3.14 |
    | 1     | Angle of 2nd joint 2 (shoulder to elbow) | Float (rad) | -3.14 | 3.14 |
    | 2     | Angle of 3rd joint 3 (elbow to wrist)    | Float (rad) | -3.14 | 3.14 |
    | 3     | Angle of 4th joint 4 (wrist to gripper)  | Float (rad) | -3.14 | 3.14 |
    | 4     | Angle of 5th joint 5 (gripper)           | Float (rad) | -3.14 | 3.14 |
    | 5     | X position of the cube                   | Float (m)   | -10.0 | 10.0 |
    | 6     | Y position of the cube                   | Float (m)   | -10.0 | 10.0 |
    | 7     | Z position of the cube                   | Float (m)   | -10.0 | 10.0 |
    | 8     | Quaternion \( w \) of the cube           | Float       | -1.0  | 1.0  |
    | 9     | Quaternion \( x \) of the cube           | Float       | -1.0  | 1.0  |
    | 10    | Quaternion \( y \) of the cube           | Float       | -1.0  | 1.0  |
    | 11    | Quaternion \( z \) of the cube           | Float       | -1.0  | 1.0  |
    | 12    | X position of the target                 | Float (m)   | -10.0 | 10.0 |
    | 13    | Y position of the target                 | Float (m)   | -10.0 | 10.0 |
    | 14    | Z position of the target                 | Float (m)   | -10.0 | 10.0 |

    ## Reward

    The reward is the negative distance between the cube and the target position. The episode is terminated when the
    distance is less than a threshold.
    """

    def __init__(self, image_state=None, action_mode="joint", render_mode=None, target_xy_range=0.2, obj_xy_range=0.2):
        super().__init__(
            xml_path=os.path.join(ASSETS_PATH, "scene_one_cube.xml"),
            image_state=image_state,
            action_mode=action_mode,
            render_mode=render_mode,
        )

        # Define the action space and observation space
        self.action_space = self.set_action_space_without_gripper()

        """
        low = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0]  # ruff: noqa: E501
        high = [np.pi, np.pi, np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]  # ruff: noqa: E501
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        """

        spaces = {
            "image_front": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(240, 320, 3)),
            "image_top": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(240, 320, 3)),
            "arm_qpos": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(5,)),
            "arm_qvel": gym.spaces.Box(low=-10.0, high=10.0, shape=(5,)),
            "object_qpos": gym.spaces.Box(low=-10.0, high=10.0, shape=(3,)),
            "object_qvel": gym.spaces.Box(low=-10.0, high=10.0, shape=(3,)),
        }
        self.observation_space = gym.spaces.Dict(spaces)

        self.threshold_distance = 0.26
        self.set_object_range(obj_xy_range)
        self.set_target_range(target_xy_range)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        self.data.qpos[:5] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Sample and set the object position
        object_pos = self.np_random.uniform(self.object_low, self.object_high).astype(np.float32)
        object_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qpos[5:12] = np.concatenate([object_pos, object_rot])

        # Sample the target position
        self.target_pos = self.np_random.uniform(self.target_low, self.target_high)

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation_dict_one_object(), {}

    def get_observation(self):
        return np.concatenate([self.data.qpos, self.target_pos], dtype=np.float32)

    def step(self, action):
        # Perform the action and step the simulation
        self.base_step_action_nograsp(action)

        # Get the new observation
        observation = self.get_observation_dict_one_object()

        # Compute the distance
        cube_id = self.model.body("box").id
        cube_pos = self.data.geom_xpos[cube_id]
        distance = np.linalg.norm(cube_pos - self.target_pos)

        # Compute the reward
        reward = -distance

        # Check if the target position is reached
        terminated = distance < self.threshold_distance

        return observation, reward, terminated, False, {}
