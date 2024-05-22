import os

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.envs.base_env import BaseRobotEnv


class StackEnv(BaseRobotEnv):
    """
    ## Description

    The robot has to stack two cubes. The episode is terminated when the blue cube is above the red cube and close to it.

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
    | 5     | X position of the red cube               | Float (m)   | -10.0 | 10.0 |
    | 6     | Y position of the red cube               | Float (m)   | -10.0 | 10.0 |
    | 7     | Z position of the red cube               | Float (m)   | -10.0 | 10.0 |
    | 8     | Quaternion \( w \) of the red cube       | Float       | -1.0  | 1.0  |
    | 9     | Quaternion \( x \) of the red cube       | Float       | -1.0  | 1.0  |
    | 10    | Quaternion \( y \) of the red cube       | Float       | -1.0  | 1.0  |
    | 11    | Quaternion \( z \) of the red cube       | Float       | -1.0  | 1.0  |
    | 12    | X position of the blue cube              | Float (m)   | -10.0 | 10.0 |
    | 13    | Y position of the blue cube              | Float (m)   | -10.0 | 10.0 |
    | 14    | Z position of the blue cube              | Float (m)   | -10.0 | 10.0 |
    | 15    | Quaternion \( w \) of the blue cube      | Float       | -1.0  | 1.0  |
    | 16    | Quaternion \( x \) of the blue cube      | Float       | -1.0  | 1.0  |
    | 17    | Quaternion \( y \) of the blue cube      | Float       | -1.0  | 1.0  |
    | 18    | Quaternion \( z \) of the blue cube      | Float       | -1.0  | 1.0  |

    ## Reward

    The reward is 1.0 if the blue cube is above the red cube and close to it, 0.0 otherwise.
    """

    def __init__(self, image_state=None, action_mode="joint", render_mode=None, obj_xy_range=0.15):
        super().__init__(
            xml_path=os.path.join(ASSETS_PATH, "scene_two_cubes.xml"),
            image_state=image_state,
            action_mode=action_mode,
            render_mode=render_mode,
        )

        # Define the action space and observation space
        self.action_space = self.set_action_space_with_gripper()
        
        low = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, -10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0]  # ruff: noqa: E501
        high = [np.pi, np.pi, np.pi, np.pi, np.pi, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]  # ruff: noqa: E501
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        self.threshold_distance = 0.05
        self.set_object_range(obj_xy_range)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        self.data.qpos[:5] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Sample and set the object position
        for obj in ["red_box_joint", "blue_box_joint"]:
            object_pos = self.np_random.uniform(self.object_low, self.object_high).astype(np.float32)
            object_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.data.joint(obj).qpos = np.concatenate([object_pos, object_rot])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        # Get the additional info
        info = self.get_info()

        return self.get_observation(), info

    def get_observation(self):
        return self.data.qpos.astype(dtype=np.float32)

    def step(self, action):
        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Get the new observation
        observation = self.get_observation()

        # Check if the stack is successful
        cube1_id = self.model.body("box1").id
        cube1_pos = self.data.geom_xpos[cube1_id]
        cube2_id = self.model.body("box2").id
        cube2_pos = self.data.geom_xpos[cube2_id]
        is_2_above_1 = cube1_pos[2] < cube2_pos[2]
        is_2_close_to_1 = np.linalg.norm(cube1_pos[0:2] - cube2_pos[0:2]) < self.threshold_distance
        success = is_2_above_1 and is_2_close_to_1

        # Compute the reward
        reward = 1.0 if success else 0.0

        terminated = success

        # Get the additional info
        info = self.get_info()

        return observation, reward, terminated, False, info
