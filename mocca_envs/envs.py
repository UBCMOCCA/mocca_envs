import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import gym.utils
import gym.utils.seeding
import numpy as np
import pybullet
import torch

from mocca_envs.bullet_utils import (
    BulletClient,
    Camera,
    SinglePlayerStadiumScene,
    VSphere,
)
from mocca_envs.robots import Cassie, Walker3D

# Hard-coding the colors to get rid of the `matplotlib` dependency
# Colors = {k: matplotlib.colors.to_rgba(v) for k, v in matplotlib.colors.cnames.items()}
Colors = {
    "dodgerblue": (0.11764705882352941, 0.5647058823529412, 1.0, 1.0),
    "crimson": (0.8627450980392157, 0.0784313725490196, 0.23529411764705882, 1.0),
}

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class EnvBase(gym.Env):
    def __init__(self, render=False):
        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.is_render = render

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self, robot_class):

        if self.physics_client_id < 0:
            self.owns_physics_client = True

            bc_mode = pybullet.GUI if self.is_render else pybullet.DIRECT
            self._p = BulletClient(connection_mode=bc_mode)

            if self.is_render:
                self.camera = Camera(self._p, 1 / self.control_step)
                if hasattr(self, "create_target"):
                    self.create_target()

            self.physics_client_id = self._p._client
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        frame_skip = 4
        self.scene = SinglePlayerStadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize()

        # Create floor
        self.ground_ids = {(self.scene.ground_plane_mjcf[0], -1)}

        # Create robot object
        self.robot = robot_class(self._p)
        self.robot.initialize()
        self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        if self.state_id < 0:
            self.state_id = self._p.saveState()

    def render(self, mode="human"):
        # Taken care of by pybullet
        pass

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self):
        keys = self._p.getKeyboardEvents()
        # keys is a dict, so need to check key exists
        if ord("d") in keys and keys[ord("d")] == self._p.KEY_WAS_RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif ord("r") in keys and keys[ord("r")] == self._p.KEY_WAS_RELEASED:
            self.done = True
        elif ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
                    break


class CassieEnv(EnvBase):

    control_step = 1 / 30
    llc_frame_skip = 50
    sim_frame_skip = 1

    ## PD gains:
    kp = np.array([100, 100, 88, 96, 98, 98, 50, 100, 100, 88, 96, 98, 98, 50])
    kd = kp / 15
    kd[[6, 13]] /= 10

    def __init__(self, render=True):
        super(CassieEnv, self).__init__(render)

        self.seed()
        self.initialize_scene_and_robot(Cassie)

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
        curr_angles = self.robot.to_radians(self.robot.joint_angles)
        curr_speeds = self.robot.joint_speeds

        perror = target_angles - curr_angles
        verror = np.clip(target_speeds - curr_speeds, -5, 5)

        # print(', '.join(['%3.0f' % s for s in perror]), end='   |   ')
        # print(', '.join(['%4.0f' % s for s in verror]))
        # import time
        # time.sleep(0.1)

        return self.kp * perror + self.kd * verror

    def step(self, a):
        target_angles = np.zeros(14)
        ## `knee_to_shin` and `ankle_joint` joints (both sides) do not have a motor
        ## we don't know how to set the constraints for them so we're using PD with fixed target instead
        target_angles[[0, 1, 2, 3, 6, 7, 8, 9, 10, 13]] = a / 2
        # target_angles = self.robot.to_radians(target_angles)
        target_angles += self.robot.base_joint_angles
        target_angles[4] = 0
        target_angles[5] = -target_angles[3] + 0.227  # -q_3 + 13 deg
        target_angles[11] = 0
        target_angles[12] = -target_angles[10] + 0.227  # -q_10 + 13 deg

        for _ in range(self.llc_frame_skip):
            target_speeds = target_angles * 0
            torque = self.pd_control(target_angles, target_speeds)
            self.robot.apply_action(torque)
            self.scene.global_step()
            robot_state = self.robot.calc_state()

        done = False
        if not np.isfinite(robot_state).all():
            print("~INF~", robot_state)
            done = True

        old_potential = self.potential
        self.potential = self.calc_potential(self.robot.body_xyz)
        progress = self.potential - old_potential

        tall_bonus = (
            2.0
            if self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2]) > 0.6
            else -1.0
        )

        if tall_bonus < 0:
            done = True

        if self.is_render:
            self.camera.track(pos=self.robot.body_xyz)
            self._handle_keyboard()
            done = done or self.done

        self.rewards = [tall_bonus, progress]

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
        return state, sum(self.rewards), done, {}

    def render(self, mode="human"):
        # Taken care of by pybullet
        self.is_render = True


class Walker3DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    def __init__(self, render=False):
        super(Walker3DCustomEnv, self).__init__(render)

        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        self.dist = 5.0

        self.seed()
        self.initialize_scene_and_robot(Walker3D)

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)

    def reset(self):
        self.done = False
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.walk_target = np.array(
            [self.dist * np.cos(angle), self.dist * np.sin(angle), 1.0]
        )
        self.close_count = 0

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(random_pose=True)

        # Reset camera
        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)

        self.calc_potential()
        state = np.concatenate(
            (
                self.robot_state,
                [self.distance_to_target * np.sin(self.angle_to_target) / self.dist],
                [self.distance_to_target * np.cos(self.angle_to_target) / self.dist],
            )
        )

        return state

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        self.robot_state = self.robot.calc_state(self.ground_ids)
        self.calc_env_state(action)

        reward = self.progress + self.target_bonus - self.energy_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        state = np.concatenate(
            (
                self.robot_state,
                [self.distance_to_target * np.sin(self.angle_to_target) / self.dist],
                [self.distance_to_target * np.cos(self.angle_to_target) / self.dist],
            )
        )

        if self.is_render:
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            if self.distance_to_target < 0.15:
                self.target.set_color(Colors["dodgerblue"])
            else:
                self.target.set_color(Colors["crimson"])

        if self.is_render:
            self._handle_keyboard()

        return state, reward, self.done, {}

    def calc_potential(self):

        walk_target_theta = np.arctan2(
            self.walk_target[1] - self.robot.body_xyz[1],
            self.walk_target[0] - self.robot.body_xyz[0],
        )
        walk_target_delta = self.walk_target - self.robot.body_xyz

        self.angle_to_target = walk_target_theta - self.robot.body_rpy[2]

        self.distance_to_target = (
            walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2
        ) ** (1 / 2)

        self.linear_potential = -self.distance_to_target / self.scene.dt
        self.angular_potential = np.cos(self.angle_to_target)

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential

        self.progress = linear_progress
        # Reward for turning at larger distances
        if self.distance_to_target > 1:
            self.progress += angular_progress

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        self.energy_penalty = self.electricity_cost * float(
            np.abs(action * self.robot.joint_speeds).mean()
        )
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())

        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        # Calculate done
        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        self.tall_bonus = 2.0 if height > 0.7 else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_target_reward(self):
        self.target_bonus = 0
        if self.distance_to_target < 0.15:
            self.close_count += 1
            self.target_bonus = 2

        if self.close_count >= 150:
            self.close_count = 0
            angle = self.np_random.uniform(-np.pi, np.pi)
            delta = self.dist * np.array([np.cos(angle), np.sin(angle), 0.0])
            self.walk_target += delta
            self.calc_potential()

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        # Order is important
        # calc_target_reward() potential
        self.calc_base_reward(action)
        self.calc_target_reward()

    def get_mirror_function(self):

        right_indices = torch.from_numpy(self.robot._right_joints).long()
        left_indices = torch.from_numpy(self.robot._left_joints).long()
        action_dim = self.robot.action_space.shape[0]

        def mirror_trajectory(trajectory_samples):
            observations_batch = trajectory_samples[0]
            states_batch = trajectory_samples[1]
            actions_batch = trajectory_samples[2]
            value_preds_batch = trajectory_samples[3]
            return_batch = trajectory_samples[4]
            masks_batch = trajectory_samples[5]
            old_action_log_probs_batch = trajectory_samples[6]
            adv_targ = trajectory_samples[7]

            def swap_lr(t, r, l):
                t[:, torch.cat((r, l))] = t[:, torch.cat((l, r))]

            # Create left / right mirrored training data
            observations_clone = observations_batch.clone()
            actions_clone = actions_batch.clone()

            # (0-6) more, (6-27) joints angles,
            # (27-48) joint speeds, (48-50) contacts, (50-52) angles
            # 2:  vy
            # 4:  roll
            # 6:  abdomen_z pos
            # 8:  abdomen_x pos
            # 27: abdomen_z vel
            # 29: abdomen_x vel
            # 50: sin(-a) = -sin(a)
            observations_clone[:, [2, 4, 6, 8, 27, 29, 50]] *= -1
            # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow, contacts
            right = right_indices.add(6)
            left = left_indices.add(6)
            right = torch.cat((right, right.add(action_dim)))  # +19 to get speeds
            left = torch.cat((left, left.add(action_dim)))
            # contacts
            right = torch.cat((right, torch.tensor([48]).long()))
            left = torch.cat((left, torch.tensor([49]).long()))
            swap_lr(observations_clone, right, left)

            # Mirror actions
            # 0, 2: abdomen_z, abdomen_x
            actions_clone[:, [0, 2]] *= -1
            # hip_[x,y,z], knee, ankle, shoulder_[x,z,y], elbow
            swap_lr(actions_clone, right_indices, left_indices)

            observations_batch = torch.cat([observations_batch, observations_clone])
            actions_batch = torch.cat([actions_batch, actions_clone])
            states_batch = states_batch.repeat((2, 1))
            value_preds_batch = value_preds_batch.repeat((2, 1))
            return_batch = return_batch.repeat((2, 1))
            masks_batch = masks_batch.repeat((2, 1))
            old_action_log_probs_batch = old_action_log_probs_batch.repeat((2, 1))
            adv_targ = adv_targ.repeat((2, 1))

            return (
                observations_batch,
                states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

        return mirror_trajectory

