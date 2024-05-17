import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym

from gym_lowcostrobot.simulated_robot import SimulatedRobot

class BaseEnv(gym.Env):

    def __init__(self, xml_path='low_cost_robot/scene_one_cube.xml', render=False, image_state=False, action_mode='joint', multi_image_state=False, max_episode_steps=200):
        super(BaseEnv, self).__init__()

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

        self.multi_image_state = multi_image_state
        self.image_state = image_state
        if self.image_state:
            self.renderer = mujoco.Renderer(self.model)


    def base_step_action_nograsp(self, action):
        if self.action_mode == 'ee':
            # Update the robot position based on the action
            ee_id = self.model.body("joint5-pad").id
            ee_target_pos = self.data.xpos[ee_id] + action[:3]

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
            q_target_pos[-1:] = 0.0 # Close the gripper
            self.robot.set_target_pos(q_target_pos)
        else:
            self.robot.set_target_pos(action)
        
        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)


    def base_step_action_withgrasp(self, action):
        if self.action_mode == 'ee':
            # Update the robot position based on the action
            ee_id = self.model.body("joint5-pad").id
            ee_target_pos = self.data.xpos[ee_id] + action[:3]

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
            q_target_pos[-1:] = np.sign(action[-1]) # Open or close the gripper
            self.robot.set_target_pos(q_target_pos)
        else:
            self.robot.set_target_pos(action)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

    def base_set_info(self, done):
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

        # Render the simulation in multiview
        if self.multi_image_state:
            dict_imgs = self.get_camera_images()
            info["dict_imgs"] = dict_imgs

        return info, done, truncated

    def reset(self):
        pass

    def step(self, action):
        pass

    def current_state(self):
        pass

    def render(self):
        if not self.do_render:
            return
        self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.step_start = time.time()
    
    def get_camera_images(self):
        dict_cams = {}
        for cam_ids in ["camera_left", "camera_right", "camera_top"]:
            self.renderer.update_scene(self.data, camera=cam_ids)
            img = self.renderer.render()
            dict_cams[cam_ids] = img
        return dict_cams
