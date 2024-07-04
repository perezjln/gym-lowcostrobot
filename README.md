# Gym Low Cost Robot

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/release/python-370/)

This repository provide comprehensive gymnasium environments for simulated applications of the [Low Cost Robot](https://github.com/AlexanderKoch-Koch/low_cost_robot). 
These environments are designed to facilitate robot learning research and development while remaining accessible and cost-effective.

https://github.com/perezjln/gym-lowcostrobot/assets/45557362/cb724171-3c0e-467f-8957-97e79eb9c852

### Features

- **Action Definitions**: The environments support both Cartesian and Joint control, allowing for diverse tasks.
- **State Definitions**: The state of the robot is defined using proprioception (joint position) and visual input from multiple cameras, enabling advanced perception capabilities on non-equipped environments.
- **Rearrangement Tasks**: A set of rearrangement tasks involving single and multiple cubic objects is provided as a starting point. These tasks are designed to help users begin experimenting with and developing robotic manipulation skills.
- **Dataset Recording**: Included is a [recording script](https://github.com/perezjln/gym-lowcostrobot/blob/main/examples/lerobotdataset_save.py) that allows users to build datasets in the `LeRobotDataset` format. This format is defined by the [🤗 Hugging Face LeRobot library](https://github.com/huggingface/lerobot), allowing to share and mutualize experiments among users and tasks.

### Goals

The primary objective of these environments is to promote end-to-end open-source and affordable robotic learning platforms. By lowering the cost and accessibility barriers, we aim to:

- **Advance Research**: Provide researchers and developers with tools to explore and push the boundaries of robotics and AI.
- **Encourage Innovation**: Foster a community of innovators who can contribute to and benefit from shared knowledge and resources.
- **Educate and Train**: Serve as a valuable educational resource for students and educators in the fields of robotics, computer science, and engineering.

By leveraging these open-source tools, we believe that more individuals, research institutions and schools can participate in and contribute to the growing field of robotic learning, ultimately driving forward the discipline as a whole.


## Installation

To install the package, use the following command:

```bash
pip install rl_zoo3
pip install git+https://github.com/perezjln/gym-lowcostrobot.git
```

## Usage

### Simulation Example: PickPlaceCube-v0

Here's a basic example of how to use the `PickPlaceCube-v0` environment in simulation:

```python
import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments

# Create the environment
env = gym.make("PickPlaceCube-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

for _ in range(1000):
    # Sample random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminted, truncated, info = env.step(action)

    # Reset the environment if it's done
    if terminted or truncated:
        observation, info = env.reset()

# Close the environment
env.close()
```

### Real-World Interface

For real-world interface with the real-world robot, just pass `simulation=False` to the environment:

```python
env = gym.make("PickPlaceCube-v0", simulation=False)
```

### Environments

Currently, the following environments are available:

- `LiftCube-v0`: Lift a cube.
- `PickPlaceCube-v0`: Pick and place a cube.
- `PushCube-v0`: Push a cube.
- `ReachCube-v0`: Reach a cube.
- `StackTwoCubes-v0`: Stack two cubes.

## Headless Mode

To run the environment in an headless machine, make sure to set the following environment variable:

```sh
export MUJOCO_GL=osmesa
export DISPLAY=:0
```

## Training Policies with Stable Baselines3 and RL Zoo3 - step-by-step guide

To train a reinforcement learning policy using Stable Baselines3 and RL Zoo3, you need to define a configuration file and then launch the training process.

### Step 1: Define a Configuration File

Create a YAML configuration file specifying the training parameters for your environment. Below is an example configuration for the `ReachCube-v0` environment:

```yaml
ReachCube-v0:
  n_timesteps: !!float 1e7
  policy: 'MultiInputPolicy'
  frame_stack: 3
  use_sde: True
```

- `n_timesteps`: The number of timesteps to train the model. Here, it is set to 10 million.
- `policy`: The policy type to be used. In this case, it is set to `'MultiInputPolicy'`.
- `frame_stack`: The number of frames to stack, which is 3 in this example.
- `use_sde`: A boolean indicating whether to use State-Dependent Exploration (SDE). It is set to `True`.

### Step 2: Launch the Training Process

After defining the configuration file, you can start the training of your policy using the following command:

```sh
python -u -m rl_zoo3.train --algo tqc --env ReachCube-v0 --gym-packages gym_lowcostrobot -conf examples/rl_zoo3_conf.yaml --env-kwargs observation_mode:'"both"' -f logs
```

- `python -u -m rl_zoo3.train`: Executes the training module from RL Zoo3.
- `--algo tqc`: Specifies the algorithm to use, in this case, TQC (Truncated Quantile Critics).
- `--env ReachCube-v0`: Specifies the environment to train on.
- `--gym-packages gym_lowcostrobot`: Includes the necessary gym packages for your environment.
- `--conf rl_zoo3_conf.yaml`: Points to the configuration file you created.
- `--env-kwargs observation_mode:'"both"'`: Passes additional environment-specific arguments.
- `-orga <huggingface_user>`: Specifies the Hugging Face organization/user where the model will be stored.
- `-f logs`: Specifies the directory where the training logs will be saved.

For more detailed information on the available options and configurations, refer to the RL Zoo3 documentation.

## Contributing

We welcome contributions to the project! Please follow these general guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Push your branch to your fork.
5. Create a pull request with a description of your changes.

Format your code with [Ruff](https://github.com/astral-sh/ruff)

```sh
ruff format gym_lowcostrobot examples tests setup.py --line-length 127
```

and test your changes with [pytest](https://docs.pytest.org/en/8.2.x/):

```sh
pytest
```

For significant changes, please open an issue first to discuss what you would like to change.

Currently, our todo list is:

- [x] Load and replay one `LeRobotDataset` in simulation
- [x] Implement inverse kinematics in each environment, improvements remain very welcome

Training:

- [ ] Train policies with SB3
- [ ] Provide reward shaping functions, get inspired from meta-world

Real-world:

- [ ] Implement controller interface for simulation, like lowcostrobot-leader
- [ ] Implement the real-world interface, seemless interface with real-world observations, motor

Simulation:

- [ ] Parameter identification
- [ ] Mesh simplification; dissociate visual and collision meshes?

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
