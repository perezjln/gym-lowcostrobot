import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

from gym_lowcostrobot.Rewards import threshold_proximity_reward

from gym_lowcostrobot.tasks.base_env import BaseEnv

class StackEnv(BaseEnv):

    def __init__(self, xml_path='low_cost_robot/scene_one_cube.xml', render=False, image_state=False, multi_image_state=False, action_mode='joint', max_episode_steps=200):
        super(StackEnv, self).__init__(xml_path=xml_path, render=render, image_state=image_state, multi_image_state=multi_image_state, action_mode=action_mode, max_episode_steps=max_episode_steps)

        # Define the action space and observation space
        self.action_mode = action_mode
        if action_mode == 'ee':
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3+1,), dtype=np.float32)
        else:
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.xpos.flatten().shape[0],), dtype=np.float32)

        # Initialize the robot and target positions
        self.target_pos = np.array([np.random.rand(), np.random.rand(), 0.1])
        self.threshold_distance = 0.05

    def reset(self, seed=42):

        self.data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]
        self.data.joint("blue_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]

        mujoco.mj_step(self.model, self.data)

        self.current_step = 0
        self.step_start = time.time()

        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
        info = {"img": img} if self.image_state else {}

        return np.concatenate([self.data.xpos.flatten()]), info

    def reward(self):
        cube1_id = self.model.body("box1").id
        cube1_pos = self.data.geom_xpos[cube1_id]

        cube2_id = self.model.body("box2").id
        cube2_pos = self.data.geom_xpos[cube2_id]

        ### simplistic version of cube stacking reward
        return threshold_proximity_reward(cube1_pos[0:2], cube2_pos[0:2], self.threshold_distance) and cube1_pos[2] < cube2_pos[2]

    def step(self, action):
        
        # Perform the action and step the simulation
        self.base_step_action_withgrasp(action)

        # Compute the reward based on the distance
        reward = self.reward()
        done = reward

        # Return the next observation, reward, done flag, and additional info
        next_observation = np.concatenate([self.data.xpos.flatten()])

        # Check if the episode is timed out, fill info dictionary
        info, done, truncated = self.base_set_info(done)

        return next_observation, float(reward), done, truncated, info

