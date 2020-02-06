# PyBullet RL Environments

"Good artists borrow, great artists steal."

This repository contains the environments used in [On Learning Symmetric Loocmotion](https://dl.acm.org/citation.cfm?id=3360070), i.e. `Walker3DCustomEnv-v0` and `Walker3DStepperEnv-v0`.  There are some other WIP environments.

## Installation
You can use the `setup.py` for installing tha `mocca_envs` package:
```bash
# Install from Github as is
pip install git+https://github.com/UBCMOCCA/mocca_envs

# Install an editable package
git clone https://github.com/UBCMOCCA/mocca_envs.git
cd mocca_envs
pip install -e .
```
You may also wish to install [PyBullet](https://pypi.org/project/pybullet/) separately.

## Environments
Current environments:
 - `CassieEnv-v0`
 - `Walker3DCustomEnv-v0`
 - `Child3DCustomEnv-v0`
 - `Walker3DChairEnv-v0`
 - `Walker3DStepperEnv-v0`
