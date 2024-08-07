import mujoco
import numpy as np


## https://alefram.github.io/posts/Basic-inverse-kinematics-in-Mujoco
# Levenberg-Marquardt method
class LevenbegMarquardtIK:
    def __init__(self, model, data, tol=0.04, step_size=0.5):
        self.model = model
        self.data = data
        self.jacp = np.zeros((3, model.nv))  # translation jacobian
        self.jacr = np.zeros((3, model.nv))  # rotational jacobian
        self.step_size = step_size
        self.tol = tol
        self.alpha = 0.5
        self.damping = 0.15

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    # Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, body_id, viewer):
        """Calculate the desire joints angles for goal"""
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        while np.linalg.norm(error) >= self.tol:
            # calculate jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)

            # calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T

            delta_q = j_inv @ error

            # compute next step
            self.data.qpos[:1] += self.step_size * delta_q

            # check limits
            # self.check_joint_limits(self.data.qpos)

            # compute forward kinematics
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()

            # calculate new error
            error = np.subtract(goal, self.data.body(body_id).xpos)
            print("Error: ", np.linalg.norm(error))


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
        # ee_pos = self.d.geom_xpos[joint_id]
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

    def inverse_kinematics_null_reg(
        self,
        ee_target_pos,
        step=0.2,
        joint_name="moving_side",
        nb_dof=6,
        regularization=1e-6,
        home_position=None,
        nullspace_weight=0.1,
    ):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :param home_position: numpy array of home joint positions to regularize towards
        :param nullspace_weight: float, weight for the nullspace regularization
        :return: numpy array of target joint positions
        """
        if home_position is None:
            home_position = np.zeros(nb_dof)  # Default to zero if no home position is provided

        try:
            # Get the joint ID from the name
            joint_id = self.m.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")

        # Get the current end effector position
        ee_id = self.m.body(joint_name).id
        ee_pos = self.d.geom_xpos[ee_id]

        # Compute the Jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        print(jac)

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos

        # Compute the pseudoinverse of the Jacobian with regularization
        jac_reg = jac[:, :nb_dof].T @ jac[:, :nb_dof] + regularization * np.eye(nb_dof)
        jac_pinv = np.linalg.inv(jac_reg) @ jac[:, :nb_dof].T

        # Compute target joint velocities
        qdot = jac_pinv @ delta_pos

        # Add nullspace regularization to keep joint positions close to the home position
        qpos = self.d.qpos[:nb_dof]
        nullspace_reg = nullspace_weight * (home_position - qpos)
        qdot += nullspace_reg

        # Normalize joint velocities to avoid excessive movements
        qdot_norm = np.linalg.norm(qdot)
        if qdot_norm > 1.0:
            qdot /= qdot_norm

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos

    def set_target_qpos(self, target_pos):
        self.d.ctrl = target_pos
