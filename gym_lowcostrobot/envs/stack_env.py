import time

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from gym_lowcostrobot.envs.base_env import BaseRobotEnv
from gym_lowcostrobot.rewards import threshold_proximity_reward


class StackEnv(BaseRobotEnv):
    def __init__(self, render=False, image_state=False, multi_image_state=False, action_mode="joint"):
        super().__init__(
            xml_path="assets/scene_two_cubes.xml",
            render=render,
            image_state=image_state,
            multi_image_state=multi_image_state,
            action_mode=action_mode,
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

        self.object_low = np.array([0.0, 0.0, 0.01])
        self.object_high = np.array([0.2, 0.2, 0.01])

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Sample the target position and set the robot position
        self.target_pos = self.np_random.uniform(self.target_low, self.target_high)

        # Set the object position
        self.data.joint("red_box_joint").qpos[:3] = self.np_random.uniform(self.object_low, self.object_high)
        self.data.joint("blue_box_joint").qpos[:3] = self.np_random.uniform(self.object_low, self.object_high)

        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        self.step_start = time.time()

        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
        info = {"img": img} if self.image_state else {}

        return self.get_observation(), info

    def get_observation(self):
        return np.concatenate([self.data.xpos.flatten()], dtype=np.float32)

    def step(self, action):
        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Compute the reward based on the distance
        cube1_id = self.model.body("box1").id
        cube1_pos = self.data.geom_xpos[cube1_id]
        cube2_id = self.model.body("box2").id
        cube2_pos = self.data.geom_xpos[cube2_id]

        # Simplistic version of cube stacking reward
        is_2_above_1 = cube1_pos[2] < cube2_pos[2]
        is_2_close_to_1 = np.linalg.norm(cube1_pos[0:2] - cube2_pos[0:2]) < self.threshold_distance
        success = is_2_above_1 and is_2_close_to_1
        reward = 1.0 if success else 0.0
        terminated = success
        observation = self.get_observation()
        info = self.get_info()
        return observation, reward, terminated, False, info
