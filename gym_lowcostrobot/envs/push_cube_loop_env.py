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
    - `"target_pos"`: the position of the target, as (x, y, z)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"target_pos"`  | ✓         | ✓         | ✓        |
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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "push_cube_loop.xml"), {})
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
        self.cube_q_id = self.model.body("cube").id

        goal_region_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_1")
        goal_region_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_region_2")

        self.goal_region_1_center = self.model.geom_pos[goal_region_1_id]
        self.goal_region_2_center = self.model.geom_pos[goal_region_2_id]

        self.goal_region_high = self.model.geom_size[goal_region_1_id] 
        self.goal_region_high[:2] -= 0.005 # offset sampling region to keep cube within
        self.goal_region_low = self.goal_region_high * np.array([-1., -1., 1.])
        self.current_goal = 0 # 0 for first goal region , and 1 for second goal region

        # indicators for the reward


    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="moving_side", nb_dof=6, regularization=1e-6):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :return: numpy array of target joint positions
        """
        try:
            # Get the joint ID from the name
            joint_id = self.model.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")

        # Get the current end effector position
        # ee_pos = self.d.geom_xpos[joint_id]
        ee_id = self.model.body(joint_name).id
        ee_pos = self.data.geom_xpos[ee_id]

        # Compute the Jacobian
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jac, None, joint_id)

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos

        # Compute the pseudoinverse of the Jacobian with regularization
        jac_reg = jac[:, :nb_dof].T @ jac[:, :nb_dof] + regularization * np.eye(nb_dof)
        jac_pinv = np.linalg.inv(jac_reg) @ jac[:, :nb_dof].T

        # Compute target joint velocities
        qdot = jac_pinv @ delta_pos

        # Normalize joint velocities to avoid excessive movements
        qdot_norm = np.linalg.norm(qdot)
        if qdot_norm > 1.0:
            qdot /= qdot_norm
  
        # Read the current joint positions
        qpos = self.data.qpos[:nb_dof]

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        if self.action_mode == "ee":
            # raise NotImplementedError("EE mode not implemented yet")
            ee_action, gripper_action = action[:3], action[-1]

            # Update the robot position based on the action
            ee_id = self.model.body("moving_side").id
            ee_target_pos = self.data.xpos[ee_id] + ee_action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
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
        cube_pos = self.np_random.uniform(self.goal_region_low, self.goal_region_high) 
        cube_pos[:2] += (1 - self.current_goal) * self.goal_region_1_center[:2] \
                      + self.current_goal * self.goal_region_2_center[:2]

        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[:] = np.concatenate([robot_qpos, cube_pos, cube_rot])

        #goal_right_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_right")
        #self.model.site_pos[goal_right_geom_id] = self.goal_right
        
        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos_xy = self.data.qpos[self.cube_q_id:self.cube_q_id+2]
        #cube_to_target = np.linalg.norm(cube_pos - self.target_pos)

        # Compute the reward
        reward = 0# -cube_to_target
        return observation, reward, False, False, {}

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

    def _get_cube_overlap(self):
        # Unpack the parameters
        x_cube, y_cube = self.cube_center[:2]
        w_cube, l_cube = self.cube_size[:2]
        
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

