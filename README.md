# Gym Low Cost Robot

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/release/python-370/)

Gymnasium environments for simulated and real-world [Low Cost Robot](https://github.com/AlexanderKoch-Koch/low_cost_robot).

https://github.com/perezjln/gym-lowcostrobot/assets/45557362/cb724171-3c0e-467f-8957-97e79eb9c852


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

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Todos
- **Improve the fidelity of the simulation**.
- **Implement inverse kinematics in each environment**.
- **Implement the real-world interface**.
- **Provide reward shaping functions**.
