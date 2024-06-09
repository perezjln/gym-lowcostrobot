import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


class BaseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, xml_path, observation_mode="image", action_mode="joint", render_mode=None):
        super().__init__()

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path, {})
        self.data = mujoco.MjData(self.model)
        self.observation_mode = observation_mode

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        if observation_mode in ["image", "both"]:
            self.renderer = mujoco.Renderer(self.model)
            self.cameras = ["image_front", "image_top"] # Image key names in observation_dict for lerobotdataset

        self.action_mode = action_mode

    def inverse_kinematics(self, ee_target_pos, step=0.2):
        """
        :param ee_target_pos: numpy array of target end effector position
        """
        joint_id = self.model.body("moving_side").id

        # Get the current end effector position
        ee_pos = self.data.geom_xpos[joint_id]

        # Compute the jacobian
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jacp=jacp, jacr=None, body=joint_id)

        # Compute target joint velocities
        qpos = self.data.qpos[:6]
        qvel = np.dot(np.linalg.pinv(jacp[:, :6]), ee_target_pos - ee_pos)

        # Apply the joint velocities
        target_qpos = qpos + qvel * step
        return target_qpos

    def set_target_qpos(self, target_pos):
        self.data.ctrl = target_pos

    def apply_action(self, action, block_gripper):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - EE mode and block gripper: [dx, dy, dz]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        - Joint mode and block gripper: [q1, q2, q3, q4, q5, q6]
        """
        if self.action_mode == "ee":
            if block_gripper:
                ee_action, gripper_action = action, 0.0
            else:
                ee_action, gripper_action = action[:3], action[-1]

            # Update the robot position based on the action
            ee_id = self.model.body("moving_side").id
            ee_target_pos = self.data.xpos[ee_id] + ee_action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
            target_qpos[-1:] = gripper_action
        elif self.action_mode == "joint":
            target_qpos = np.zeros(6, dtype=np.float32)
            target_qpos[:5] = action[:5]
            if block_gripper:
                target_qpos[-1] = 0.0
            else:
                target_qpos[-1] = action[-1]
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.set_target_qpos(target_qpos)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

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
            geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
            object1_id = geom_names.index("red_box")
            observation["object_qpos"] = self.data.geom_xpos[object1_id].astype(np.float32)
            if "blue_box" in geom_names:
                object2_id = geom_names.index("blue_box")
                observation["object2_qpos"] = self.data.geom_xpos[object2_id].astype(np.float32)
        return observation

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
            time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.step_start = time.time()
