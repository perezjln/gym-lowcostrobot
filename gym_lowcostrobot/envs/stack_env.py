import time
import numpy as np

import mujoco
import mujoco.viewer

from gymnasium import spaces

from gym_lowcostrobot.rewards import threshold_proximity_reward

from gym_lowcostrobot.envs.base_env import BaseEnv


class StackEnv(BaseEnv):
    def __init__(
        self,
        xml_path="assets/scene_two_cubes.xml",
        image_state=None,
        action_mode="joint",
        render_mode=None
    ):
        super().__init__(
            xml_path=xml_path,
            image_state=image_state,
            action_mode=action_mode,
            render_mode=render_mode
        )

        # Define the action space and observation space
        if self.action_mode == "ee":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3 + 1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.xpos.flatten().shape[0],),
            dtype=np.float32,
        )

        self.threshold_distance = 0.05

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Sample the target position and set the robot position
        self.target_pos = np.array([self.np_random.random(), self.np_random.random(), 0.1])
        self.data.joint("red_box_joint").qpos[:3] = [
            self.np_random.random() * 0.2,
            self.np_random.random() * 0.2,
            0.01,
        ]
        self.data.joint("blue_box_joint").qpos[:3] = [
            self.np_random.random() * 0.2,
            self.np_random.random() * 0.2,
            0.01,
        ]

        mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        self.step_start = time.time()

        info = self.get_info()

        return np.concatenate([self.data.xpos.flatten()], dtype=np.float32), info

    def reward(self):
        cube1_id = self.model.body("box1").id
        cube1_pos = self.data.geom_xpos[cube1_id]

        cube2_id = self.model.body("box2").id
        cube2_pos = self.data.geom_xpos[cube2_id]

        ### simplistic version of cube stacking reward
        return (
            threshold_proximity_reward(cube1_pos[0:2], cube2_pos[0:2], self.threshold_distance)
            and cube1_pos[2] < cube2_pos[2]
        )

    def step(self, action):
        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Compute the reward based on the distance
        reward = self.reward()
        terminated = reward

        # Return the next observation, reward, terminated flag, and additional info
        next_observation = np.concatenate([self.data.xpos.flatten()], dtype=np.float32)

        # Check if the episode is timed out, fill info dictionary
        info = self.get_info()

        return next_observation, float(reward), terminated, False, info
