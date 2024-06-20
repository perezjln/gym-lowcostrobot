import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_lowcostrobot import ASSETS_PATH


class LiftCubeEnv(Env):
    """
    ## Description

    The robot has to lift a cube with its end-effector.

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

    The reward is the sum of two terms: the height of the cube above the threshold and the negative distance between the
    end effector and the cube.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "lift_cube.xml"), {})
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4}[action_mode]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

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
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = -75
            self.viewer.cam.distance = 1
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)

        # Set additional utils
        self.threshold_height = 0.5
        self.cube_low = np.array([-0.15, 0.10, 0.015])
        self.cube_high = np.array([0.15, 0.25, 0.015])

    def inverse_kinematics(self, ee_target_pos):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :return: numpy array of target joint positions
        """
        home_position = np.zeros(6)  # Default to zero if no home position is provided
        joint_id = self.model.body("moving_side").id

        error_tolerance = 1e-2  # error tolerance for the end effector position
        max_iterations = 10  # maximum number of iterations
        step = 0.05  # step size for the iteration
        regularization = 1e-6  # regularization factor for the pseudoinverse computation
        nullspace_weight = 1.0  # weight for the nullspace regularization

        ee_pos = self.data.xpos[joint_id]
        error = ee_target_pos - ee_pos
        jac = np.zeros((3, self.model.nv))
        q_pos = self.data.qpos[7:13].copy()
        Kn = np.ones(6) * nullspace_weight

        num_iterations = 0
        while np.linalg.norm(error) > error_tolerance and num_iterations < max_iterations:
            # Compute the Jacobian
            # mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_jac(self.model, self.data, jac, None, ee_target_pos, joint_id)
            ee_pos = self.data.xpos[joint_id].astype(np.float32)

            # Compute the difference between target and current end effector positions
            error = ee_target_pos - ee_pos

            # Compute the pseudoinverse of the Jacobian with damping, nv has 12 values
            jac_reg = jac[:, 6:12].T @ jac[:, 6:12] + regularization * np.eye(6)
            jac_pinv = np.linalg.inv(jac_reg) @ jac[:, 6:12].T

            # Compute target joint velocities
            qdot = jac_pinv @ error
            # try to keep the joint close to home position, otherwise the robot will move even if the target is reached
            qdot += (np.eye(6) - np.linalg.pinv(jac[:, 6:12]) @ jac[:, 6:12]) @ (Kn * (home_position - q_pos))

            # Normalize joint velocities to avoid excessive movements
            qdot_norm = np.linalg.norm(qdot)
            if qdot_norm > 1.0:
                qdot /= qdot_norm

            # Update the joint positions
            self.data.qpos[7:13] += qdot * step
            num_iterations += 1

        q_target_pos = self.data.qpos[7:13].copy()
        self.data.qpos[7:13] = q_pos
        return q_target_pos

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        if self.action_mode == "ee":
            ee_action, gripper_action = action[:3], action[-1]

            # Update the robot position based on the action
            ee_id = self.model.body("moving_side").id
            ee_target_pos = self.data.xpos[ee_id] + ee_action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos)
            target_qpos[-1:] = gripper_action
        elif self.action_mode == "joint":
            target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            target_qpos = action * (target_high - target_low) / 2 + (target_high + target_low) / 2
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)
        if self.render_mode == "human":
            self.viewer.sync()

    def get_observation(self):
        # qpos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, q6, gripper]
        # qvel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dq6, dgripper]
        observation = {
            "arm_qpos": self.data.qpos[7:13].astype(np.float32),
            "arm_qvel": self.data.qvel[6:12].astype(np.float32),
        }
        if self.observation_mode in ["image", "both"]:
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["image_front"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        if self.observation_mode in ["state", "both"]:
            observation["cube_pos"] = self.data.qpos[:3].astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position and sample the cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[:] = np.concatenate([cube_pos, cube_rot, robot_qpos])

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos = self.data.qpos[:3]
        cube_z = cube_pos[2]
        ee_id = self.model.body("moving_side").id
        ee_pos = self.data.xpos[ee_id]
        ee_to_cube = np.linalg.norm(ee_pos - cube_pos)

        # Compute the reward
        reward_height = cube_z - self.threshold_height
        reward_distance = -ee_to_cube
        reward = reward_height + reward_distance
        info = {}
        return observation, reward, False, False, info

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
