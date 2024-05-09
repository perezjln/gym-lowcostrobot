import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

from envs.interface import SimulatedRobot


class ReachCubeEnv(gym.Env):

    def __init__(self, xml_path='low_cost_robot/scene_one_cube.xml', render=False, image_state=False):
        super(ReachCubeEnv, self).__init__()

        # Load the MuJoCo model and data
        self.model  = mujoco.MjModel.from_xml_path(xml_path)
        self.data   = mujoco.MjData(self.model)
        self.robot  = SimulatedRobot(self.model, self.data)

        self.do_render = render
        if self.do_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        self.image_state = image_state
        if self.image_state:
            self.renderer = mujoco.Renderer(self.model)

        # Define the action space and observation space
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.xpos.flatten().shape[0] + 3,), dtype=np.float32)

        # Initialize the robot and target positions
        self.threshold_distance = 0.01

    def reset(self):
        self.step_start = time.time()

        self.data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]
        mujoco.mj_step(self.model, self.data)

        box_id = self.model.body("box").id
        return np.concatenate([self.data.xpos.flatten(), self.data.xpos[box_id]])

    def reward(self):
        cube_id = self.model.body("box").id
        cube_pos = self.data.geom_xpos[cube_id]
        ee_id = self.model.body("joint5-pad").id
        ee_pos = self.data.geom_xpos[ee_id]
        return np.linalg.norm(cube_pos - ee_pos)

    def step(self, action):

        # Update the robot position based on the action
        ee_id = self.model.body("joint5-pad").id
        ee_target_pos = self.data.xpos[ee_id] + action

        # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
        q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
        q_target_pos[-1:] = 0.0 # Close the gripper
        self.robot.set_target_pos(q_target_pos)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)
        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()

        # Compute the reward based on the distance
        distance = self.reward()
        reward   = -distance

        # Check if the target position is reached
        done = distance < self.threshold_distance

        # Return the next observation, reward, done flag, and additional info
        cube_id = self.model.body("box").id
        cube_pos = self.data.geom_xpos[cube_id]
        next_observation = np.concatenate([self.data.xpos.flatten(), cube_pos])
        
        return next_observation, reward, done, {"img": img} if self.image_state else {}

    def render(self):
        if not self.do_render:
            return
        self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.step_start = time.time()


def do_env_sim():
    
    import matplotlib.pyplot as plt

    do_log = False
    env = ReachCubeEnv(render=False, image_state=True)
    env.reset()

    max_step = 1000
    for _ in range(max_step):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        plt.imshow(info["img"]) 
        plt.show()

        if do_log:
            print("Observation:", observation)
            print("Reward:", reward)

        if done:
            print("Cube reached the target position")
            env.reset()

        env.render()


if __name__ == '__main__':
    do_env_sim()