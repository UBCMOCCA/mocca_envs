import os
import gym
import pickle
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Cassie, Cassie2D
from mocca_envs import current_dir


class CassieEnv(EnvBase):

    control_step = 1 / 30
    llc_frame_skip = 50
    sim_frame_skip = 1

    ## PD gains:
    kp = np.array(
        [
            100,
            100,
            88,
            96,
            # 98,
            # 98,
            50,
            #
            100,
            100,
            88,
            96,
            # 98,
            # 98,
            50,
        ]
    )
    kd = kp / 15
    # kd[[6, 13]] /= 10

    def __init__(self, render=False, planar=False):
        robot_class = Cassie2D if planar else Cassie
        super(CassieEnv, self).__init__(robot_class, render)

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def calc_potential(self, body_xyz):
        target_dist = (
            (self.walk_target[1] - body_xyz[1]) ** 2
            + (self.walk_target[0] - body_xyz[0]) ** 2
        ) ** (1 / 2)

        return -target_dist / self.control_step

    def reset(self):
        self.done = False
        self.walk_target = np.array([1000.0, 0.0, 0.0])

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset()

        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)

        self.potential = self.calc_potential(self.robot.body_xyz)
        delta = self.walk_target - self.robot.body_xyz
        walk_target_theta = np.arctan2(delta[1], delta[0])
        delta_theta = walk_target_theta - self.robot.body_rpy[2]
        rot = np.array(
            [
                [np.cos(-delta_theta), -np.sin(-delta_theta), 0.0],
                [np.sin(-delta_theta), np.cos(-delta_theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        target = np.matmul(rot, self.walk_target)
        state = np.concatenate((self.robot_state, target[0:2]))
        return state

    def pd_control(self, target_angles, target_speeds):
        curr_angles = self.robot.to_radians(self.robot.joint_angles)[
            self.robot.powered_joint_inds
        ]
        curr_speeds = self.robot.joint_speeds[self.robot.powered_joint_inds]

        perror = target_angles - curr_angles
        verror = np.clip(target_speeds - curr_speeds, -5, 5)

        # print(', '.join(['%3.0f' % s for s in perror]), end='   |   ')
        # print(', '.join(['%4.0f' % s for s in verror]))
        # import time
        # time.sleep(0.1)

        return self.kp * perror + self.kd * verror

    def base_angles(self):
        return np.array(self.robot.base_joint_angles)

    def compute_rewards(self, action, torques):
        old_potential = self.potential
        self.potential = self.calc_potential(self.robot.body_xyz)
        progress = self.potential - old_potential

        tall_bonus = (
            2.0
            if self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2]) > 0.6
            else -1.0
        )

        dead = tall_bonus < 0

        return dead, {"AliveRew": tall_bonus, "ProgressRew": progress}

    def step(self, a):
        target_angles = self.base_angles()[self.robot.powered_joint_inds]
        ## `knee_to_shin` and `ankle_joint` joints (both sides) do not have a motor
        ## we don't know how to set the constraints for them so we're using PD with fixed target instead
        target_angles += a
        # target_angles = self.robot.to_radians(target_angles)
        # target_angles +=   # self.robot.base_joint_angles
        # target_angles[4] = 0
        # target_angles[5] = -target_angles[3] + 0.227  # -q_3 + 13 deg
        # target_angles[11] = 0
        # target_angles[12] = -target_angles[10] + 0.227  # -q_10 + 13 deg

        torques = []

        for _ in range(self.llc_frame_skip):
            target_speeds = target_angles * 0
            torque = self.pd_control(target_angles, target_speeds)
            torques.append(torque)
            self.robot.apply_action(torque)
            self.scene.global_step()
            robot_state = self.robot.calc_state()

        done = False
        if not np.isfinite(robot_state).all():
            print("~INF~", robot_state)
            done = True

        dead, rewards = self.compute_rewards(a, torques)
        done = done or dead

        if self.is_render:
            self.camera.track(
                pos=self.robot.body_xyz, smooth_coef=np.array([0.1, 0.01, 0.01])
            )
            self._handle_keyboard()
            done = done or self.done

        delta = self.walk_target - self.robot.body_xyz
        walk_target_theta = np.arctan2(delta[1], delta[0])
        delta_theta = walk_target_theta - self.robot.body_rpy[2]

        rot = np.array(
            [
                [np.cos(-delta_theta), -np.sin(-delta_theta), 0.0],
                [np.sin(-delta_theta), np.cos(-delta_theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        target = np.matmul(rot, self.walk_target)
        state = np.concatenate((robot_state, target[0:2]))
        return state, sum(rewards.values()), done, rewards


class CassieMocapEnv(CassieEnv):
    mocap_path = os.path.join(current_dir, "data/cassie/mocap/", "cassie_step_data.pkl")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.mocap_path, "rb") as datafile:
            self.step_data = pickle.load(datafile)

    def reset(self):
        self.phase = 0
        return super().reset()

    def base_angles(self):
        return self.step_data[self.phase * 60]  # TODO: fix time

    def compute_rewards(self, action, torques):
        dead, rewards = super(CassieMocapEnv, self).compute_rewards(action, torques)
        rewards["AliveRew"] /= 2
        rewards["ProgressRew"] /= 3
        rewards["EnergyUseRew"] = -0.0002 * np.mean(np.power(torques, 2))
        rewards["DeviationRew"] = -0.2 * np.mean(np.power(action, 2))

        dyn_angles = self.robot.to_radians(self.robot.joint_angles)
        kin_angles = self.base_angles()

        joint_penalty = np.sqrt(
            np.sum((kin_angles - dyn_angles)[self.robot.powered_joint_inds] ** 2)
        )
        rewards["ImitationRew"] = 2 * np.exp(-4 * joint_penalty)

        return dead, rewards

    def step(self, action):
        self.phase = (self.phase + 1) % 28
        return super().step(action)

