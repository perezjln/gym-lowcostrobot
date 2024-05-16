import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

from envs.tasks.BaseEnv import BaseEnv

class LiftCubeEnv(BaseEnv):

    def __init__(self, xml_path='low_cost_robot/scene_one_cube.xml', render=False, image_state=False, multi_image_state=False, action_mode='joint', max_episode_steps=200):
        super(LiftCubeEnv, self).__init__(xml_path=xml_path, render=render, image_state=image_state, multi_image_state=multi_image_state, action_mode=action_mode, max_episode_steps=max_episode_steps)

        # Define the action space and observation space
        self.action_mode = action_mode
        if action_mode == 'ee':
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3+1,), dtype=np.float32)
        else:
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.xpos.flatten().shape[0],), dtype=np.float32)

        # Initialize the robot and target positions
        self.target_pos = np.array([np.random.rand(), np.random.rand(), 0.1])
        self.threshold_distance = 0.5

    def reset(self, seed=42):
        self.data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]
        mujoco.mj_step(self.model, self.data)
        self.step_start = time.time()
        self.current_step = 0

        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
        info = {"img": img} if self.image_state else {}

        return self.current_state(), info

    def reward(self):
        cube_id = self.model.body("box").id
        cube_pos = self.data.geom_xpos[cube_id]
        return cube_pos[-1]

    def current_state(self):
        return np.concatenate([self.data.xpos.flatten()])

    def step(self, action):

        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Compute the reward based on the distance
        high = self.reward()

        # Check if the target position is reached
        done = high > self.threshold_distance

        # Return the next observation, reward, done flag, and additional info
        next_observation = self.current_state()
        
        # Check if the episode is timed out, fill info dictionary
        info, done, truncated = self.base_set_info(done)
            
        return next_observation, high, done, truncated, info

