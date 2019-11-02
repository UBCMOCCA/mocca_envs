# PyBullet RL Environments

"Good artists borrow, great artists steal."

Sorry, the documentation is pretty much non-existent at this point. I'll try to work on it in the next couple of days. The `master` branch is also way behind the other ones which should be fixed soon. [`symmetric_rl`](https://github.com/UBCMOCCA/mocca_envs/tree/symmetric-rl) is the branch that was used in our recent publication "On Learning Symmetric Loocmotion".

## Installation
You can use the `setup.py` for installing tha `mocca_envs` package:
```bash
## You might need to use `python3` or prepend `sudo` depending on your setup 
python setup.py install
```
You may also wish to install **PyBullet** manually.

## Environments
Current environments:
 - `CassieEnv-v0`
 - `Walker3DCustomEnv-v0`
 - `Child3DCustomEnv-v0`
 - `Walker3DChairEnv-v0`
 - `Walker3DStepperEnv-v0`
