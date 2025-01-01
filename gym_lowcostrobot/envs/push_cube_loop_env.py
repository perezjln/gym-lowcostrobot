import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_lowcostrobot import ASSETS_PATH


class PushCubeLoopEnv(Env):
    """
    ## Description

    The robot has to push a cube with its end-effector between two goal positions.
    Once the cube reaches the goal position, the goal region is switched to the other side.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"cube_pos"`    |           | ✓         | ✓        |

    ## Reward

    The reward is the negative distance between the cube and the target position.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        observation_mode="image",
        action_mode="joint",
        block_gripper=True,
        n_substeps=20,
        render_mode=None,
    ):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "push_cube_loop.xml"))
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        self.block_gripper = block_gripper
        action_shape = {"joint": 5, "ee": 3}[self.action_mode]
        action_shape += 0 if self.block_gripper else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        self.num_dof = 6

        # Set the observations space
        self.observation_mode = observation_mode
        observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "arm_qvel": spaces.Box(low=-10.0, high=10.0, shape=(6,)),
        }
        if self.observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model)
        if self.observation_mode in ["state", "both"]:
            observation_subspaces["cube_pos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        # Set the render utilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.azimuth = -45.0
            self.viewer.cam.distance = 1.5
            self.viewer.cam.elevation = -20.0
            self.viewer.cam.lookat = np.array([0.0, 0.0, 0.0])
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)

        self.cube_size = 0.015 / 2
        self.cube_position = np.array([0.0, 0.0, 0.0])

        goal_region_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_1")
        goal_region_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_2")

        self.goal_region_1_center = self.model.geom_pos[goal_region_1_id]
        self.goal_region_2_center = self.model.geom_pos[goal_region_2_id]

        self.goal_region_high = self.model.geom_size[goal_region_1_id] / 2
        self.goal_region_high[:2] -= 0.008  # offset sampling region to keep cube within
        self.goal_region_low = self.goal_region_high * np.array([-1.0, -1.0, 1.0])
        self.current_goal = 0  # 0 for first goal region , and 1 for second goal region
        self.control_decimation = n_substeps  # number of simulation steps per control step

        self._step = 0
        # indicators for the reward

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

        return q

    def inverse_kinematics(
        self,
        ee_target_pos,
        ee_site="end_effector_site",
        num_dof=6,
        step=0.5,
        lm_damping=0.15,
        max_iter=10,
        tolerance_err=0.01,
        home_position=None,
        nullspace_weight=0.0,
    ):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param ee_site: str, name of the end effector site
        :param num_dof: int, number of degrees of freedom
        :param step: float, step size for the iteration
        :param lm_damping: float, regularization factor for the pseudoinverse computation
        :param max_iter: int, maximum number of iterations
        :param tolerance_err: float, tolerance error
        :param home_position: numpy array of home joint positions to regularize towards
        :param nullspace_weight: float, weight for the nullspace regularization
        :return: numpy array of target joint positions
        """

        if home_position is None:
            home_position = np.zeros(num_dof)  # Default to zero if no home position is provided

        jacp = np.zeros((3, self.model.nv))
        ee_id = self.model.site(ee_site).id

        # Initial joint positions
        q = self.data.qpos[:num_dof].copy()

        for _ in range(max_iter):
            self.data.qpos[:num_dof] = q
            mujoco.mj_forward(self.model, self.data)

            ee_pos = self.data.site(ee_id).xpos
            error = ee_target_pos - ee_pos
            error_norm = np.linalg.norm(error)

            # Stop iterations
            if error_norm < tolerance_err:
                break

            # Jacobian
            mujoco.mj_jacSite(self.model, self.data, jacp, None, ee_id)

            # Damped least squares (Levenberg-Marquardt Algorithm)
            jac_reg = jacp[:, :num_dof].T @ jacp[:, :num_dof] + lm_damping * np.eye(num_dof)
            jac_pinv = np.linalg.inv(jac_reg) @ jacp[:, :num_dof].T
            qdot = jac_pinv @ error

            # Nullspace control biasing joint velocities towards the home configuration
            qdot += (np.eye(num_dof) - np.linalg.pinv(jacp[:, :num_dof]) @ jacp[:, :num_dof]) @ (
                nullspace_weight * (home_position - self.data.qpos[:num_dof])
            )

            # Normalize joint velocity
            qdot_norm = np.linalg.norm(qdot)
            if qdot_norm > 1.0:
                qdot /= qdot_norm

            # Compute the new joint positions. Integrate joint velocities to obtain joint positions.
            q += qdot * step

            # Check limits
            q = self.check_joint_limits(q)

        q_target_pos = q
        return q_target_pos

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, gripper]
        """

        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.action_mode == "ee":
            ee_action = action[:3]

            # Update the robot position based on the action
            ee_id = self.model.site("end_effector_site").id
            ee_target_pos = self.data.site(ee_id).xpos + ee_action * 0.05  # limit maximum change in position
            ee_target_pos[2] = np.max((0, ee_target_pos[2]))

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
            # Block the gripper for push task
            target_qpos[-1:] = np.array([0])
        elif self.action_mode == "joint":
            target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            # Separate arm joint angles and gripper position
            current_joint_angles = self.data.qpos[: self.num_dof].copy()
            arm_action = np.array(action[: self.num_dof - 1])
            # Block the gripper for push task
            gripper_action = np.array([0])

            # Clip and update arm joint positions
            target_arm_qpos = np.clip(
                arm_action + current_joint_angles[: self.num_dof - 1],
                target_low[: self.num_dof - 1],
                target_high[: self.num_dof - 1],
            )

            # Clip and update gripper position
            target_gripper_pos = gripper_action

            # Combine arm and gripper targets
            target_qpos = np.append(target_arm_qpos, target_gripper_pos)
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()

    def get_observation(self):
        # qpos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, gripper]
        # qvel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dgripper]
        observation = {
            "arm_qpos": self.data.qpos[: self.num_dof].astype(np.float32),
            "arm_qvel": self.data.qvel[: self.num_dof].astype(np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["image_front"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        if self.observation_mode in ["state", "both"]:
            observation["cube_pos"] = self.data.qpos[self.num_dof : self.num_dof + 3].astype(np.float32).copy()
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position and sample the cube position
        cube_pos = self.np_random.uniform(self.goal_region_low, self.goal_region_high)
        cube_pos[:2] += (1 - self.current_goal) * self.goal_region_1_center[
            :2
        ] + self.current_goal * self.goal_region_2_center[:2]

        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[: self.num_dof] = robot_qpos
        self.data.qpos[self.num_dof : self.num_dof + 7] = np.concatenate([cube_pos, cube_rot])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {"timestamp": 0.0}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        reward, success = self.get_reward()
        self._step += 1
        info = {"timestamp": self.data.time, "success": success}  # self.model.opt.timestep * self._step}
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def get_reward(self):
        # Get the position of the cube and the distance between the end effector and the cube
        self.cube_position = self.data.qpos[self.num_dof : self.num_dof + 3].astype(np.float32).copy()
        overlap = self.get_cube_overlap()
        # if the intersection is above 95% consider the episode a success and switch goals:
        success = 0
        if overlap > 0.95:
            success = 1
            reward = +5
            self.current_goal = 1 - self.current_goal

        elif overlap > 0.0:
            reward = overlap - 1

        elif overlap == 0.0:
            # calculate distance to edge on y axis only
            goal_region_edge = (
                self.goal_region_low[1]
                + (1 - self.current_goal) * self.goal_region_1_center[1]
                + self.current_goal * self.goal_region_2_center[1]
            )

            distance_to_edge = np.sqrt((self.cube_position[1] - goal_region_edge) ** 2)
            # max distance to edge within the box is 0.16
            reward = min(max((-distance_to_edge / 0.16) - 1, -2), -1)
        return reward, success

    def get_cube_overlap(self):
        # Unpack the parameters
        x_cube, y_cube = self.cube_position[:2]
        w_cube = l_cube = self.cube_size

        goal_center = self.goal_region_1_center if self.current_goal == 0 else self.goal_region_2_center
        x_goal, y_goal = goal_center[:2]
        w_goal, l_goal = self.goal_region_high[:2]

        # Calculate the overlap along the x-axis
        x_overlap = max(0, min(x_cube + w_cube, x_goal + w_goal) - max(x_cube - w_cube, x_goal - w_goal))

        # Calculate the overlap along the y-axis
        y_overlap = max(0, min(y_cube + l_cube, y_goal + l_goal) - max(y_cube - l_cube, y_goal - l_goal))

        # Calculate the area of the overlap region
        overlap_area = x_overlap * y_overlap

        # Calculate the area of the cube
        cube_area = w_cube * l_cube * 4

        # return the percentage overlap relative to the cube area
        return overlap_area / cube_area

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.observation_mode in ["image", "both"]:
            self.renderer.close()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()
