import mujoco
import numpy as np


class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        """
        :param pos: numpy array of joint positions in range [-pi, pi]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return (pos / 3.14 + 1.0) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self, nb_dof=5) -> np.ndarray:
        """
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:nb_dof]

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name="end_effector"):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        return self.d.geom_xpos[joint_id]

    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="end_effector", nb_dof=5):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]

        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)

        # compute target joint velocities
        qpos = self.read_position(nb_dof=nb_dof)
        qdot = np.dot(np.linalg.pinv(jac[:, :nb_dof]), ee_target_pos - ee_pos)

        # apply the joint velocities
        q_target_pos = qpos + qdot * step
        return q_target_pos

    def inverse_kinematics_reg(self, ee_target_pos, step=0.2, joint_name="end_effector", nb_dof=5, regularization=1e-6):
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
            joint_id = self.m.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")
        
        # Get the current end effector position
        #ee_pos = self.d.geom_xpos[joint_id]
        ee_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[ee_id]

        # Compute the Jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)

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
        qpos = self.read_position(nb_dof=nb_dof)

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos
