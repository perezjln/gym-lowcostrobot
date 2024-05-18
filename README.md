# gym-lowcostrobot

This repository contains the code for gymnasium environments for the lowcostrobot project.
https://github.com/AlexanderKoch-Koch/low_cost_robot/tree/main

Various task associated with diverse action and observation modalities will be available.

Simple inverse kinematics
https://github.com/perezjln/envs-lowcostrobot/assets/5373778/de8c6448-1ece-4823-89ee-ad59d05a431d


## Installation

```
cd gym-lowcostrobot
pip install .
```

## Test

```
pip install pytest
pytest
```

## Format

```
pip install ruff
ruff format gym_lowcostrobot examples tests setup.py --line-length 119
isort -l 119 .
```
