import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments
import numpy as np

def displace_object(square_size=0.15, invert_y=False, origin_pos=[0, 0.1]):
    ### Sample a position in a square in front of the robot
    if not invert_y:
        x = np.random.uniform(origin_pos[0] - square_size / 2, origin_pos[0] + square_size / 2)
        y = np.random.uniform(origin_pos[1] - square_size / 2, origin_pos[1] + square_size / 2)
    else:
        x = np.random.uniform(origin_pos[0] + square_size / 2, origin_pos[0] - square_size / 2)
        y = np.random.uniform(origin_pos[1] + square_size / 2, origin_pos[1] - square_size / 2)
    env.data.qpos[:3] = np.array([x, y, origin_pos[2]])
    return env.data.qpos[:3]

# Create the environment
env = gym.make("ReachCube-v0", render_mode="human", action_mode="ee")

# Reset the environment
observation, info = env.reset()
cube_origin_pos = env.data.qpos[:3].astype(np.float32)
for i in range(10000):
    if i % 500 == 0:
        cube_pos = displace_object(square_size=0.2, invert_y=False, origin_pos=cube_origin_pos)
    # Sample random action
    action = env.action_space.sample()
    ee_id = env.model.body("moving_side").id
    ee_pos = env.data.xpos[ee_id].astype(np.float32) # default [0.03390873 0.22571199 0.14506643]
    action[:3] = cube_pos + [0,0,0.1] - ee_pos
    # action[:3] = [0.03390873, 0.22571199, 0.14506643]
    action[3] = -1.5

    # Step the environment
    observation, reward, terminted, truncated, info = env.step(action)

    # Reset the environment if it's done
    if terminted or truncated:
        observation, info = env.reset()

# Close the environment
env.close()