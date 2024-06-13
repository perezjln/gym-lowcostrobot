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
- **Dataset Recording**: Included is a [recording script](https://github.com/perezjln/gym-lowcostrobot/blob/main/examples/trace_lerobotdataset.py) that allows users to build datasets in the `LeRobotDataset` format. This format is defined by the [🤗 Hugging Face LeRobot library](https://github.com/huggingface/lerobot), allowing to share and mutualize experiments among users and tasks.

### Goals

The primary objective of these environments is to promote end-to-end open-source and affordable robotic learning platforms. By lowering the cost and accessibility barriers, we aim to:

- **Advance Research**: Provide researchers and developers with tools to explore and push the boundaries of robotics and AI.
- **Encourage Innovation**: Foster a community of innovators who can contribute to and benefit from shared knowledge and resources.
- **Educate and Train**: Serve as a valuable educational resource for students and educators in the fields of robotics, computer science, and engineering.

By leveraging these open-source tools, we believe that more individuals, research institutions and schools can participate in and contribute to the growing field of robotic learning, ultimately driving forward the discipline as a whole.


## Installation

To install the package, use the following command:

```bash
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

- [ ] Load and replay one theRobotDataset in simulation.
- [ ] Implement controller interface for simulation, like lowcostrobot-leader
- [ ] Implement inverse kinematics in each environment.
- [ ] Implement the real-world interface, seemless interface with real-world observations, motor.
- [ ] Improve the fidelity of the simulation.
- [ ] Provide reward shaping functions.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
