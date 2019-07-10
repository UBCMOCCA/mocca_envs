import gym
import time
import numpy as np

import mocca_envs

env_name = "CassieMocapEnv-v0"
env = gym.make(env_name, render=True)

obs = env.reset()

for i in range(1000):
    env.unwrapped.phase = i % 28
    env.unwrapped.resetJoints()
    env.unwrapped._handle_keyboard()

    time.sleep(1 / 30)

