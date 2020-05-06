import os

current_dir = os.path.dirname(os.path.realpath(__file__))

from mocca_envs.walker3d_envs import Walker3DStepperEnv
from mocca_envs.robots import Mike

class MikeStepperEnv(Walker3DStepperEnv):
    robot_class = Mike
