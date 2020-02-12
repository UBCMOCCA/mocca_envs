import os

from gym.spaces import Box

current_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class CartArm2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_xpos = np.zeros(3)
        self.time_limit = 200
        self.time_tick = 0

        mujoco_env.MujocoEnv.__init__(self, os.path.join(current_dir, "data", "custom", "cartarm2d.xml"), 5)
        utils.EzPickle.__init__(self)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low[1:], high=high[1:], dtype=np.float32)
        return self.action_space

    def reset_target(self):
        target_xpos = np.concatenate([np.random.uniform(-5, 5, 1), [0], np.random.uniform(0, 2, 1)])
        self.sim.data.set_mocap_pos("targetball", target_xpos)
        self.time_tick = 0

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[0:2] = ctrl[0]
        self.sim.data.ctrl[2:] = ctrl[1:]
        for _ in range(n_frames):
            self.sim.step()

    def step(self, action):
        # target_idx = self.model.body_names.index("targetball")
        # self.sim.data.xfrc_applied[target_idx] = np.array([0, 0, 9.8, 0, 0, 0])
        self.time_tick += 1
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        delta_xpos = self.get_delta_xpos()
        delta_xpos_ss = np.sum(delta_xpos ** 2)
        reward = np.exp(-delta_xpos_ss)
        done = False

        if delta_xpos_ss <= 1e-1:
            reward += 10
            self.reset_target()

        if self.time_tick >= self.time_limit:
            done = True

        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat
        qvel = self.sim.data.qvel.flat
        delta_xpos = self.get_delta_xpos()
        return np.concatenate([qpos[1:], qvel, delta_xpos])

    def get_delta_xpos(self):
        hand_xpos = self.sim.data.get_body_xpos("hand")
        target_xpos = self.sim.data.get_mocap_pos("targetball")
        delta_xpos = hand_xpos - target_xpos
        return delta_xpos

    def reset_model(self):
        self.reset_target()
        qpos = self.init_qpos
        qvel = self.init_qvel
        # qvel[3:] += np.random.uniform(-0.3, 0.3, self.init_qvel.shape)[3:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
