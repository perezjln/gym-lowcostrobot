import numpy as np


### End effector space rewards
def proximity_reward(pos_current, pos_target):
    return -np.linalg.norm(pos_current - pos_target)


def progressive_proximity_reward(old_pos_current, old_pos_target, pos_current, pos_target):
    return np.linalg.norm(old_pos_current - old_pos_target) - np.linalg.norm(pos_current - pos_target)


def threshold_proximity_reward(pos_current, pos_target, distance_threshold):
    distance = np.linalg.norm(pos_current - pos_target)
    return 1.0 if distance < distance_threshold else 0.0


### Joint space rewards
def joint_limit_reward(joint_angles, joint_limits):
    return -np.sum(np.abs(joint_angles) > joint_limits)


def joint_penalty_reward(joint_angles, joint_penalty):
    return -np.sum(np.abs(joint_angles) * joint_penalty)


def joint_velocity_penalty_reward(joint_velocities, joint_velocity_penalty):
    return -np.sum(np.abs(joint_velocities) * joint_velocity_penalty)


def joint_acceleration_penalty_reward(joint_accelerations, joint_acceleration_penalty):
    return -np.sum(np.abs(joint_accelerations) * joint_acceleration_penalty)
