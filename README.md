# gym-lowcostrobot

https://github.com/perezjln/gym-lowcostrobot/assets/45557362/cb724171-3c0e-467f-8957-97e79eb9c852

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
