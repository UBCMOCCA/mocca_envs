# PyBullet RL Environments

"Good artists borrow, great artists steal."

This repository contains the environments used in [On Learning Symmetric Loocmotion](https://dl.acm.org/citation.cfm?id=3360070), i.e. `Walker3DCustomEnv-v0` and `Walker3DStepperEnv-v0`. There are some other WIP environments.

## Installation

You can use `setup.py` to install the `mocca_envs` package:

```bash
# Install from Github as is
pip install git+https://github.com/UBCMOCCA/mocca_envs

# Install an editable package
git clone https://github.com/UBCMOCCA/mocca_envs.git
cd mocca_envs
pip install -e .
```

You may also wish to install [PyBullet](https://pypi.org/project/pybullet/) separately.

### Testing

```bash
# Start Monkey3D
python mocca_envs/test_env.py

# See `mocca_envs/__init__.py` for list of envs
python mocca_envs/test_env.py mocca_envs:<env>
```

## Projects

The following projects use `mocca_envs`.

- On Learning Symmetric Loocmotion.
  [[Paper](https://dl.acm.org/citation.cfm?id=3360070)] [[Code](https://github.com/UBCMOCCA/SymmetricRL)]
- ALLSTEPS: Curriculum-driven Learning of Stepping Stone Skills. [[Paper](https://dl.acm.org/doi/abs/10.1111/cgf.14115)] [[Code](https://github.com/belinghy/SteppingStone)]
