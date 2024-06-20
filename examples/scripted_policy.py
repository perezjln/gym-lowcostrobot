import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np
import mujoco


def displace_object(env, square_size=0.15, invert_y=False, origin_pos=[0, 0.1]):
    ### Sample a position in a square in front of the robot
    if not invert_y:
        x = np.random.uniform(origin_pos[0] - square_size / 2, origin_pos[0] + square_size / 2)
        y = np.random.uniform(origin_pos[1] - square_size / 2, origin_pos[1] + square_size / 2)
    else:
        x = np.random.uniform(origin_pos[0] + square_size / 2, origin_pos[0] - square_size / 2)
        y = np.random.uniform(origin_pos[1] + square_size / 2, origin_pos[1] - square_size / 2)
    env.data.qpos[:3] = np.array([x, y, origin_pos[2]])
    return env.data.qpos[:3]


class BasePolicy:
    def __init__(self, inject_noise=False, init_pose=None, meet_pose=None):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None
        self.right_trajectory = None
        self.init_pose = init_pose
        self.meet_pose = meet_pose

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(self.init_pose, self.meet_pose)

        # obtain arm waypoints
        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        next_waypoint = self.trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            xyz = xyz + np.random.uniform(-scale, scale, xyz.shape)

        action = np.concatenate([xyz, quat, [gripper]])

        self.step_count += 1
        return action


class LiftCubePolicy(BasePolicy):

    def generate_trajectory(self, init_pose, meet_pose):
        init_pose = init_pose
        meet_pose = meet_pose

        # box_info = np.array(ts_first.observation['env_state'])
        # box_xyz = box_info[:3]
        # box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_xyz = meet_pose[:3]

        self.trajectory = [
            {"t": 0, "xyz": init_pose[:3], "quat": init_pose[3:], "gripper": 0}, # sleep
            {"t": 800, "xyz": meet_xyz + np.array([0.0, 0.01, 0.075]), "quat": meet_pose[3:], "gripper": -1.5}, # approach meet position
            {"t": 1100, "xyz": meet_xyz + np.array([0.0, 0.01, 0.02]), "quat": meet_pose[3:], "gripper": -1.5}, # move to meet position
            {"t": 1350, "xyz": meet_xyz + np.array([0.0, 0.01, 0.02]), "quat": meet_pose[3:], "gripper": -0.25}, # close gripper
            {"t": 1490, "xyz": meet_xyz + np.array([0.0, 0.01, 0.1]), "quat": meet_pose[3:], "gripper": -0.25}, # lift up
            {"t": 1500, "xyz": meet_xyz + np.array([0.0, 0.01, 0.1]), "quat": meet_pose[3:], "gripper": -0.25}, # stay
        ]


def test_policy(task_name):

    # setup the environment
    if 'LiftCube' in task_name:
        env = gym.make("LiftCube-v0", render_mode="human", action_mode="ee")
    # other tasks can be added here
    else:
        raise NotImplementedError
    
    NUM_EPISODES = 5
    cube_origin_pos = [0.03390873, 0.22571199, 0.04]

    for episode_idx in range(NUM_EPISODES):
        observation, info = env.reset()
        cube_pos = displace_object(env, square_size=0.1, invert_y=False, origin_pos=cube_origin_pos)
        # cube_pos = env.unwrapped.data.qpos[:3].astype(np.float32)
        ee_id = env.model.body("moving_side").id
        ee_pos = env.unwrapped.data.xpos[ee_id].astype(np.float32) # default [0.03390873 0.22571199 0.14506643]
        ee_orn = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_orn, env.unwrapped.data.xmat[ee_id])
        # keep orientation constant
        init_pose = np.concatenate([ee_pos, ee_orn])
        meet_pose = np.concatenate([cube_pos, ee_orn])
        policy = LiftCubePolicy(init_pose=init_pose, meet_pose=meet_pose)
        episode_length = 1500
        for i in range(episode_length):
            action = env.action_space.sample()
            result = policy()
            ee_pos = env.unwrapped.data.xpos[ee_id].astype(np.float32)
            action[:3] = result[:3] - ee_pos
            action[3] = result[7]
            # Step the environment
            observation, reward, terminted, truncated, info = env.step(action)

            # Reset the environment if it's done
            if terminted or truncated:
                observation, info = env.reset()
                break


if __name__ == '__main__':
    test_task_name = 'LiftCube'
    test_policy(test_task_name)
