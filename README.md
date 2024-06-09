# gym-lowcostrobot

This repository contains the code for gymnasium environments for a collection of lowcost-robots including https://github.com/AlexanderKoch-Koch/low_cost_robot/tree/main

Various task associated with diverse action and observation modalities will be available.

Simple inverse kinematics
https://github.com/perezjln/envs-lowcostrobot/assets/5373778/de8c6448-1ece-4823-89ee-ad59d05a431d


## Installation

```sh
cd gym-lowcostrobot
pip install .
```

## Test

```sh
pip install pytest
pytest
```

## Format

```sh
pip install ruff
ruff format gym_lowcostrobot examples tests setup.py --line-length 119
isort -l 119 gym_lowcostrobot examples tests setup.py
```

To use headless:

```sh
export MUJOCO_GL=osmesa
export DISPLAY=:0
```

Train 
python -m rl_zoo3.train --algo ddpg --env LiftCube-v0 --env-kwargs observation_mode:'"state"' --gym-package gym_lowcostrobot

python -m rl_zoo3.push_to_hub --algo ppo --env LiftCube-v0 -orga qgallouedec -f logs --load-best