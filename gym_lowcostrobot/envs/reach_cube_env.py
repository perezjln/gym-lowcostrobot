import os

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.envs.base_env import BaseRobotEnv


class ReachCubeEnv(BaseRobotEnv):
    """
    ## Description

    The robot has to reach a cube with its end-effector. The episode is terminated when the end-effector is within a
    threshold distance from the cube.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 5-dimensional box
    representing the target joint angles.

    | Index | Action                      | Type (unit) | Min  | Max |
    | ----- | --------------------------- | ----------- | ---- | --- |
    | 0     | Joint 1 (base to shoulder)  | Float (rad) | -1.0 | 1.0 |
    | 1     | Joint 2 (shoulder to elbow) | Float (rad) | -1.0 | 1.0 |
    | 2     | Joint 3 (elbow to wrist)    | Float (rad) | -1.0 | 1.0 |
    | 3     | Joint 4 (wrist to gripper)  | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 3-dimensional box representing the target end-effector position.

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

    ## Reward

    The reward is the negative distance between the end-effector and the cube. The episode is terminated when the
    distance is less than a threshold.
    """

    def __init__(self, image_state=None, action_mode="joint", render_mode=None, obj_xy_range=0.15):
        super().__init__(
            xml_path=os.path.join(ASSETS_PATH, "scene_one_cube.xml"),
            image_state=image_state,
            action_mode=action_mode,
            render_mode=render_mode,
        )

        # Define the action space and observation space
        self.action_space = self.set_action_space_without_gripper()

        low = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0])
        high = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, 10.0, 10.0, 10.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize the robot and target positions
        self.threshold_distance = 0.01
        self.set_object_range(obj_xy_range)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        self.data.qpos[:5] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Sample and set the object position
        object_pos = self.np_random.uniform(self.object_low, self.object_high).astype(np.float32)
        object_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qpos[5:12] = np.concatenate([object_pos, object_rot])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        # Get the additional info
        info = self.get_info()

        return self.get_observation(), info

    def get_observation(self):
        return self.data.qpos[:8].astype(np.float32)

    def step(self, action):
        
        # Perform the action and step the simulation
        self.base_step_action_nograsp(action)

        # Get the new observation
        observation = self.get_observation()

        # Compute the distance between the cube and the end-effector
        cube_pos = self.data.joint("red_box_joint").qpos[:3]
        ee_pos = self.data.joint("joint5").qpos[:3]
        distance = np.linalg.norm(cube_pos - ee_pos)

        # Compute the reward based on the distance
        reward = -distance

        # The episode is terminated if the distance is less than the threshold
        terminated = distance < self.threshold_distance

        # Get the additional info
        info = self.get_info()

        return observation, reward, terminated, False, info
