import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import numpy as np

import mocca_envs

env_name = "CassieEnv-v0"
env = gym.make(env_name, render=True)
action_dim = env.action_space.shape[0]
offset = 6

# Disable gravity
# env.unwrapped._p.setGravity(0, 0, 0)

obs = env.reset()

while True:
    ## uncomment to drive the base/standing position as the action instead
    # to_normalized = env.unwrapped.robot.to_normalized
    # base_angles = env.unwrapped.robot.base_joint_angles
    # base_pose_action = to_normalized(base_angles)[[0,1,2,3,6, 7,8,9,10,13]]
    # obs, reward, done, info = env.step(base_pose_action)

    obs, reward, done, info = env.step(env.action_space.sample())

    if done:
        env.reset()
