import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym

from gym_lowcostrobot.simulated_robot import SimulatedRobot


class BaseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        xml_path,
        render=False,
        image_state=False,
        action_mode="joint",
        multi_image_state=False,
        render_mode=None,
    ):
        super().__init__()

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.do_render = render
        if self.do_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        self.multi_image_state = multi_image_state
        self.image_state = image_state
        if self.image_state:
            self.renderer = mujoco.Renderer(self.model)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_mode = action_mode

    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="end_effector"):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.model.body(joint_name).id

        # get the current end effector position
        ee_pos = self.data.geom_xpos[joint_id]

        # compute the jacobian
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jac, None, joint_id)

        # compute target joint velocities
        qpos = self.data.qpos[:5]
        qdot = np.dot(np.linalg.pinv(jac[:, :5]), ee_target_pos - ee_pos)

        # apply the joint velocities
        q_target_pos = qpos + qdot * step
        return q_target_pos

    def set_target_pos(self, target_pos):
        self.data.ctrl = target_pos

    def base_step_action_nograsp(self, action):
        if self.action_mode == "ee":
            # Update the robot position based on the action
            ee_id = self.model.body("joint5-pad").id
            ee_target_pos = self.data.xpos[ee_id] + action[:3]

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
            q_target_pos[-1:] = 0.0  # Close the gripper
            self.set_target_pos(q_target_pos)
        else:
            self.set_target_pos(action)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

    def base_step_action_withgrasp(self, action):
        if self.action_mode == "ee":
            # Update the robot position based on the action
            ee_id = self.model.body("joint5-pad").id
            ee_target_pos = self.data.xpos[ee_id] + action[:3]

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
            q_target_pos[-1:] = np.sign(action[-1])  # Open or close the gripper
            self.set_target_pos(q_target_pos)
        else:
            self.set_target_pos(action)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

    def get_info(self):
        if self.image_state:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()

        # Check if the episode is timed out
        info = {"img": img} if self.image_state else {}

        # Render the simulation in multiview
        if self.multi_image_state:
            dict_imgs = self.get_camera_images()
            info["dict_imgs"] = dict_imgs

        return info

    def render(self):
        if self.do_render:
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
