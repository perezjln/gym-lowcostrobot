import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

from envs.SimulatedRobot import SimulatedRobot


class PickPlaceCubeEnv(gym.Env):

    def __init__(self, xml_path='low_cost_robot/scene_one_cube.xml', render=False, image_state=False, action_mode='joint', max_episode_steps=200):
        super(PickPlaceCubeEnv, self).__init__()

        # Load the MuJoCo model and data
        self.model  = mujoco.MjModel.from_xml_path(xml_path)
        self.data   = mujoco.MjData(self.model)
        self.robot  = SimulatedRobot(self.model, self.data)

        self.current_step = 0
        self.max_episode_steps = max_episode_steps

        self.do_render = render
        if self.do_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        self.image_state = image_state
        if self.image_state:
            self.renderer = mujoco.Renderer(self.model)

        # Define the action space and observation space
        self.action_mode = action_mode
        if action_mode == 'ee':
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3+1,), dtype=np.float32)
        else:
            self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.xpos.flatten().shape[0] + 3,), dtype=np.float32)

        # Initialize the robot and target positions
        self.target_pos = np.array([np.random.rand(), np.random.rand(), 0.1])
        self.threshold_distance = 0.26

    def reset(self, seed=42):

        self.target_pos = np.array([np.random.rand()*0.2, np.random.rand()*0.2, 0.01])
        self.data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]

        mujoco.mj_step(self.model, self.data)

        self.step_start = time.time()
        self.current_step = 0

        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
        info = {"img": img} if self.image_state else {}

        return np.concatenate([self.data.xpos.flatten(), self.target_pos]), info

    def reward(self):
        cube_id = self.model.body("box").id
        cube_pos = self.data.geom_xpos[cube_id]
        return np.linalg.norm(cube_pos - self.target_pos)

    def step(self, action):
        
        if self.action_mode == 'ee':
            # Update the robot position based on the action
            ee_id = self.model.body("joint5-pad").id
            ee_target_pos = self.data.xpos[ee_id] + action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
            q_target_pos[-1:] = np.sign(action[-1]) # Open or close the gripper
            self.robot.set_target_pos(q_target_pos)
        else:
            self.robot.set_target_pos(action)
            
        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

        # Compute the reward based on the distance
        distance = self.reward()
        reward = -distance

        # Check if the target position is reached
        done = distance < self.threshold_distance

        # Return the next observation, reward, done flag, and additional info
        next_observation = np.concatenate([self.data.xpos.flatten(), self.target_pos])
        
        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()

        # Check if the episode is timed out
        info = {"img": img} if self.image_state else {}
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
            truncated = True
            info['TimeLimit.truncated'] = True

        return next_observation, reward, done, truncated, info

    def render(self):
        if not self.do_render:
            return
        self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.step_start = time.time()

