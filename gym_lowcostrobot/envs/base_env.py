import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


class BaseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, xml_path, observation_mode="image", action_mode="joint", render_mode=None):
        super().__init__()

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        if observation_mode in ["image", "both"]:
            self.renderer = mujoco.Renderer(self.model)

        self.set_fps(self.metadata["render_fps"])

        self.action_mode = action_mode
        self.current_step = 0

    def set_action_space_with_gripper(self):
        # Define the action space and observation space
        if self.action_mode == "ee":
            low_action = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high_action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            action_size = 3 + 1
        else:
            low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            action_size = 6
        return spaces.Box(low=low_action, high=high_action, shape=(action_size,), dtype=np.float32)

    def set_action_space_without_gripper(self):
        # Define the action space and observation space
        if self.action_mode == "ee":
            low_action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            high_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            action_size = 3
        else:
            low_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            action_size = 5
        return spaces.Box(low=low_action, high=high_action, shape=(action_size,), dtype=np.float32)

      
    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="gripper_opening", nb_joint=6):
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
        qpos = self.data.qpos[:nb_joint]
        qdot = np.dot(np.linalg.pinv(jac[:, :nb_joint]), ee_target_pos - ee_pos)

        # apply the joint velocities
        q_target_pos = qpos + qdot * step
        return q_target_pos

    def set_target_pos(self, target_pos):
        self.data.ctrl = target_pos

    def get_actuator_ranges(self):
        return self.model.actuator_ctrlrange

    def base_step_action_nograsp(self, action, joint_name="moving_side"):
        if self.action_mode == "ee":
            # Update the robot position based on the action
            ee_id = self.model.body(joint_name).id
            ee_target_pos = self.data.xpos[ee_id] + action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name=joint_name)
            q_target_pos[-1:] = 0.0  # Close the gripper
            self.set_target_pos(q_target_pos)
        else:
            q_target_pos = np.zeros(action.shape[0] + 1)
            q_target_pos[:-1] = action
            self.set_target_pos(q_target_pos)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

    def base_step_action_withgrasp(self, action, joint_name="moving_side"):
        if self.action_mode == "ee":
            # Update the robot position based on the action
            ee_id = self.model.body(joint_name).id
            ee_target_pos = self.data.xpos[ee_id] + action[:3]

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            q_target_pos = self.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name=joint_name)
            q_target_pos[-1:] = np.sign(action[-1])  # Open or close the gripper
            self.set_target_pos(q_target_pos)
        else:
            self.set_target_pos(action)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

    def set_fps(self, fps):
        if self.render_mode:
            self.model.opt.timestep = 1 / fps

    def render(self):
        if self.render_mode is not None:
            self.viewer.sync()
            time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.step_start = time.time()

    def get_camera_images(self):
        dict_cams = {}
        for cam_ids in ["camera_front", "camera_top"]:
            self.renderer.update_scene(self.data, camera=cam_ids)
            img = self.renderer.render()
            dict_cams[cam_ids] = img
        return dict_cams

    def set_object_range(self, obj_xy_range):
        self.object_low = np.array([-obj_xy_range, -obj_xy_range, 0.05])
        self.object_high = np.array([obj_xy_range, obj_xy_range, 0.05])

    def set_target_range(self, target_xy_range):
        self.target_low = np.array([-target_xy_range, -target_xy_range, 0.05])
        self.target_high = np.array([target_xy_range, target_xy_range, 0.05])

    def get_observation_dict_one_object(self):
        observation = {
            "arm_qpos": self.data.qpos[:6].astype(np.float32),
            "arm_qvel": self.data.qvel[:6].astype(np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            dict_imgs = self.get_camera_images()
            observation["image_front"] = dict_imgs["camera_front"]
            observation["image_top"] = dict_imgs["camera_top"]
        if self.observation_mode in ["state", "both"]:
            observation["object_qpos"] = self.data.qpos[6:9].astype(np.float32)
            observation["object_qvel"] = self.data.qvel[6:9].astype(np.float32)

    def get_observation_dict_two_objects(self):
        raise NotImplementedError()

