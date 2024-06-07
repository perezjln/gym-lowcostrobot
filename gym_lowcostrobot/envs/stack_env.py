import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from gym_lowcostrobot import ASSETS_PATH
from gym_lowcostrobot.envs.base_env import BaseRobotEnv


class StackEnv(BaseRobotEnv):
    """
    ## Description

    The robot has to stack two cubes. The episode is terminated when the blue cube is above the red cube and close
    to it.

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
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"object1_qpos"`: the position of the first cube, as (x, y, z)
    - `"object2_qpos"`: the position of the second cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key              | `"image"` | `"state"` | `"both"` |
    | ---------------- | --------- | --------- | -------- |
    | `"arm_qpos"`     | ✓         | ✓         | ✓        |
    | `"arm_qvel"`     | ✓         | ✓         | ✓        |
    | `"image_front"`  | ✓         |           | ✓        |
    | `"image_top"`    | ✓         |           | ✓        |
    | `"object1_qpos"` |           | ✓         | ✓        |
    | `"object2_qpos"` |           | ✓         | ✓        |

    ## Reward

    The reward is 1.0 if the blue cube is above the red cube and close to it, 0.0 otherwise.
    """

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None):
        super().__init__(
            xml_path=os.path.join(ASSETS_PATH, "scene_two_cubes.xml"),
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
            # "target_qpos": spaces.Box(low=-10.0, high=10.0, shape=(3,)),
        }
        if observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
        if observation_mode in ["state", "both"]:
            observation_subspaces["object1_qpos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
            observation_subspaces["object2_qpos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        self.threshold_distance = 0.01
        self.object_low = np.array([-0, 15, -0, 15, 0.05])
        self.object_high = np.array([0, 15, 0, 15, 0.05])

    def get_observation(self):
        observation = {
            "arm_qpos": self.data.qpos[:6].astype(np.float32),
            "arm_qvel": self.data.qvel[:6].astype(np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["image_front"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        if self.observation_mode in ["state", "both"]:
            cube1_id = self.model.body("box").id
            observation["object1_qpos"] = self.data.geom_xpos[cube1_id].astype(np.float32)
            cube2_id = self.model.body("box_two").id
            observation["object2_qpos"] = self.data.geom_xpos[cube2_id].astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        self.data.qpos[:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Sample and set the object position
        for obj in ["red_box_joint", "blue_box_joint"]:
            object_pos = self.np_random.uniform(self.object_low, self.object_high).astype(np.float32)
            object_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.data.joint(obj).qpos = np.concatenate([object_pos, object_rot])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Check if the stack is successful
        cube1_id = self.model.body("box").id
        cube1_pos = self.data.geom_xpos[cube1_id]
        cube2_id = self.model.body("box_two").id
        cube2_pos = self.data.geom_xpos[cube2_id]
        is_2_above_1 = cube1_pos[2] < cube2_pos[2]
        is_2_close_to_1 = np.linalg.norm(cube1_pos[0:2] - cube2_pos[0:2]) < self.threshold_distance
        success = is_2_above_1 and is_2_close_to_1

        # Compute the reward
        reward = 1.0 if success else 0.0

        terminated = success

        return observation, reward, terminated, False, {}
