import math

import gym
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_objects import (
    VSphere,
    Pillar,
    Plank,
    LargePlank,
    Chair,
    Bench,
    HeightField,
)
from mocca_envs.robots import Child3D, Walker3D


Colors = {
    "dodgerblue": (0.11764705882352941, 0.5647058823529412, 1.0, 1.0),
    "crimson": (0.8627450980392157, 0.0784313725490196, 0.23529411764705882, 1.0),
}

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class Walker3DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Walker3D
    termination_height = 0.7

    def __init__(self, **kwargs):
        super().__init__(self.robot_class, **kwargs)
        self.robot.set_base_pose(pose="running_start")
        self.eval_mode = False
        self.random_pose = True

        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)

    def randomize_target(self):
        if self.eval_mode:
            self.dist = 4
            self.angle = 0
        else:
            self.dist = self.np_random.uniform(3, 5)
            self.angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self.stop_frames = self.np_random.choice([30.0, 60.0])

    def evaluation_mode(self):
        self.eval_mode = True

    def reset(self):
        self.done = False
        self.add_angular_progress = True
        self.randomize_target()

        self.walk_target = np.array(
            [self.dist * math.cos(self.angle), self.dist * math.sin(self.angle), 1.0]
        )
        self.close_count = 0

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(random_pose=self.random_pose)

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)

        self.calc_potential()

        sin_ = self.distance_to_target * math.sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * math.cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))

        state = np.concatenate((self.robot_state, [sin_], [cos_]))

        return state

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        if self.eval_mode:
            self.walk_target = np.array([self.robot.body_xyz[0] + 4, 0, 1.0])

        self.robot_state = self.robot.calc_state(self.ground_ids)
        self.calc_env_state(action)

        reward = self.progress + self.target_bonus - self.energy_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        sin_ = self.distance_to_target * math.sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * math.cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))

        state = np.concatenate((self.robot_state, [sin_], [cos_]))

        if self.is_rendered or self.use_egl:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            self.target.set_color(
                Colors["dodgerblue"]
                if self.distance_to_target < 0.15
                else Colors["crimson"]
            )

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
        self.angular_potential = math.cos(self.angle_to_target)

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential

        self.calc_potential()

        if self.distance_to_target < 1:
            self.add_angular_progress = False

        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential

        self.progress = linear_progress
        # if self.add_angular_progress:
        #     self.progress += 100 * angular_progress

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
        self.tall_bonus = 2.0 if height > self.termination_height else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_target_reward(self):
        self.target_bonus = 0
        if self.distance_to_target < 0.15:
            self.close_count += 1
            self.target_bonus = 2

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        # Order is important
        # calc_target_reward() potential
        self.calc_base_reward(action)
        self.calc_target_reward()

        if self.close_count >= self.stop_frames:
            self.close_count = 0
            self.add_angular_progress = True
            self.randomize_target()
            delta = self.dist * np.array(
                [math.cos(self.angle), math.sin(self.angle), 0.0]
            )
            self.walk_target += delta
            self.calc_potential()

    def get_mirror_indices(self):

        action_dim = self.robot.action_space.shape[0]
        # _ + 6 accounting for global
        right = self.robot._right_joint_indices + 6
        # _ + action_dim to get velocities, 48 is right foot contact
        right = np.concatenate((right, right + action_dim, [48]))
        # Do the same for left, except using 49 for left foot contact
        left = self.robot._left_joint_indices + 6
        left = np.concatenate((left, left + action_dim, [49]))

        # Used for creating mirrored observations
        # 2:  vy
        # 4:  roll
        # 6:  abdomen_z pos
        # 8:  abdomen_x pos
        # 27: abdomen_z vel
        # 29: abdomen_x vel
        # 50: sin(-a) = -sin(a)
        negation_obs_indices = np.array([2, 4, 6, 8, 27, 29, 50], dtype=np.int64)
        right_obs_indices = right
        left_obs_indices = left

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        right_action_indices = self.robot._right_joint_indices
        left_action_indices = self.robot._left_joint_indices

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )


class Child3DCustomEnv(Walker3DCustomEnv):

    robot_class = Child3D
    termination_height = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.robot.set_base_pose(pose="crawl")

    def calc_base_reward(self, action):
        super().calc_base_reward(action)


class Walker3DChairEnv(Walker3DCustomEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.robot.set_base_pose(pose="sit")

    def create_terrain(self):

        self.chair = Chair(self._p)
        self.bench = Bench(self._p, pos=np.array([3, 0, 0]))

    def randomize_target(self):
        self.dist = 3.0
        self.angle = 0
        self.stop_frames = 1000

    def reset(self):
        self.done = False
        self.add_angular_progress = True
        self.randomize_target()

        self.walk_target = np.array(
            [self.dist * math.cos(self.angle), self.dist * math.sin(self.angle), 1.0]
        )
        self.close_count = 0

        self._p.restoreState(self.state_id)

        # Disable random pose for now
        # How to set on chair with random pose?
        self.robot.reset(random_pose=False, pos=(0, 0, 1.1))
        self.robot_state = self.robot.calc_state()

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)

        self.calc_potential()

        sin_ = self.distance_to_target * math.sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * math.cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))

        state = np.concatenate((self.robot_state, [sin_], [cos_]))

        return state


class Walker3DStepperEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    # Pillar, Plank, LargePlank
    plank_class = LargePlank

    def __init__(self, **kwargs):

        # Need these before calling constructor
        # because they are used in self.create_terrain()
        self.step_radius = 0.25
        self.rendered_step_count = 4
        self.stop_frames = 30

        super().__init__(Walker3D, remove_ground=True, **kwargs)
        self.robot.set_base_pose(pose="running_start")

        # Robot settings
        self.terminal_height = 0.7
        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1
        self.random_start = True

        # Env settings
        self.n_steps = 24
        self.lookahead = 2
        self.next_step_index = 0

        # Fix-ordered Curriculum
        self.curriculum = 0
        self.max_curriculum = 9

        # Terrain info
        self.dist_range = np.array([0.75, 1.2])
        self.pitch_range = np.array([-25, +25])  # degrees
        self.yaw_range = np.array([-0, 0])
        self.tilt_range = np.array([-0, 0])
        # x, y, z, phi, x_tilt, y_tilt
        self.terrain_info = np.zeros((self.n_steps, 6))

        # robot_state + (2 targets) * (x, y, z, x_tilt, y_tilt)
        high = np.inf * np.ones(
            self.robot.observation_space.shape[0] + self.lookahead * 5
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def generate_step_placements(self):

        # Check just in case
        self.curriculum = min(self.curriculum, self.max_curriculum)
        ratio = self.curriculum / self.max_curriculum

        # {self.max_curriculum + 1} levels in total
        dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1)
        dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
        yaw_range = self.yaw_range * ratio * DEG2RAD
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        N = self.n_steps
        dr = self.np_random.uniform(*dist_range, size=N)
        dphi = self.np_random.uniform(*yaw_range, size=N)
        dtheta = self.np_random.uniform(*pitch_range, size=N)
        x_tilt = self.np_random.uniform(*tilt_range, size=N)
        y_tilt = self.np_random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1:3] = 0.75
        dphi[1:3] = 0.0
        dtheta[1:3] = np.pi / 2

        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)

        dx = dr * np.sin(dtheta) * np.cos(dphi)
        dy = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        # Fix overlapping steps
        dx_max = np.maximum(np.abs(dx[2:]), self.step_radius * 2.5)
        dx[2:] = np.sign(dx[2:]) * np.minimum(dx_max, self.dist_range[1])

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        return np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

    def create_terrain(self):

        self.steps = []
        step_ids = set()
        cover_ids = set()

        for index in range(self.rendered_step_count):
            p = self.plank_class(self._p, self.step_radius)
            self.steps.append(p)
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids)

        if not self.remove_ground:
            self.all_contact_object_ids |= self.ground_ids

    def randomize_terrain(self):
        self.terrain_info = self.generate_step_placements()
        for index in range(self.rendered_step_count):
            pos = self.terrain_info[index, 0:3]
            phi, x_tilt, y_tilt = self.terrain_info[index, 3:6]
            quaternion = np.array(self._p.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
            self.steps[index].set_position(pos=pos, quat=quaternion)

    def update_steps(self):
        threshold = int(self.rendered_step_count // 2)
        if self.next_step_index >= threshold:
            oldest = (self.next_step_index - threshold - 1) % self.rendered_step_count

            next = min(
                (self.next_step_index - threshold - 1) + self.rendered_step_count,
                len(self.terrain_info) - 1,
            )
            pos = self.terrain_info[next, 0:3]
            phi, x_tilt, y_tilt = self.terrain_info[next, 3:6]
            quaternion = np.array(self._p.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
            self.steps[oldest].set_position(pos=pos, quat=quaternion)

    def reset(self):
        self.done = False
        self.target_reached_count = 0

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(
            random_pose=self.random_start, pos=[0.3, 0, 1.32]
        )
        self.calc_feet_state()

        # Randomize platforms
        self.randomize_terrain()
        self.next_step_index = 0

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        # Order is important because walk_target is set up above
        self.calc_potential()

        state = np.concatenate((self.robot_state, self.targets.flatten()))

        return state

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)

        reward = self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus - self.speed_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        state = np.concatenate((self.robot_state, self.targets.flatten()))

        if self.is_rendered or self.use_egl:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            self.target.set_color(
                Colors["dodgerblue"]
                if self.distance_to_target < 0.15
                else Colors["crimson"]
            )

        return state, reward, self.done, {}

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)

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

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        self.progress = linear_progress

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        v = self.robot.body_vel
        speed = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** (1 / 2)
        self.speed_penalty = max(speed - 1.6, 0)

        self.energy_penalty = self.electricity_cost * float(
            np.abs(action * self.robot.joint_speeds).mean()
        )
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())

        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        self.tall_bonus = 2.0 if height > self.terminal_height else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]
        target_cover_id = {(next_step.id, next_step.cover_id)}

        self.foot_dist_to_target = np.array([0.0, 0.0])

        p_xyz = self.terrain_info[self.next_step_index, [0, 1, 2]]
        self.target_reached = False
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())

            # if contact_ids is not empty, then foot is in contact
            self.robot.feet_contact[i] = 1.0 if contact_ids else 0.0

            delta = self.robot.feet_xyz[i] - p_xyz
            distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
            self.foot_dist_to_target[i] = distance

            if target_cover_id & contact_ids:
                self.target_reached = True

        # At least one foot is on the plank
        if self.target_reached:
            self.target_reached_count += 1

            # Make target stationary for a bit
            if self.target_reached_count >= self.stop_frames:
                self.next_step_index += 1
                self.target_reached_count = 0
                self.update_steps()

            # Prevent out of bound
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1

    def calc_step_reward(self):

        self.step_bonus = 0
        if (
            self.target_reached
            and self.target_reached_count == 1
            and self.next_step_index != len(self.terrain_info) - 1  # exclude last step
        ):
            self.step_bonus = 50 * 2.718 ** (-self.foot_dist_to_target.min() / 0.25)

        # For last step only
        self.target_bonus = 0
        if (
            self.next_step_index == len(self.terrain_info) - 1
            and self.distance_to_target < 0.15
        ):
            self.target_bonus = 2.0

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        cur_step_index = self.next_step_index

        # detects contact and set next step
        self.calc_feet_state()
        self.calc_base_reward(action)
        self.calc_step_reward()
        # use next step to calculate next k steps
        self.targets = self.delta_to_k_targets(k=self.lookahead)

        if cur_step_index != self.next_step_index:
            self.calc_potential()

    def delta_to_k_targets(self, k=1):
        """ Return positions (relative to root) of target, and k-1 step after """
        targets = self.terrain_info[self.next_step_index : self.next_step_index + k]
        if len(targets) < k:
            # If running out of targets, repeat last target
            targets = np.concatenate(
                (targets, np.repeat(targets[[-1]], k - len(targets), axis=0))
            )

        self.walk_target = targets[[1], 0:3].mean(axis=0)

        delta_pos = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.linalg.norm(delta_pos[:, 0:2], ord=2, axis=1)

        deltas = np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,  # x
                np.cos(angle_to_targets) * distance_to_targets,  # y
                delta_pos[:, 2],  # z
                targets[:, 4],  # x_tilt
                targets[:, 5],  # y_tilt
            ),
            axis=1,
        )

        return deltas

    def get_mirror_indices(self):

        action_dim = self.robot.action_space.shape[0]

        right_obs_indices = np.concatenate(
            (
                # joint angle indices + 6 accounting for global
                6 + self.robot._right_joint_indices,
                # joint velocity indices
                6 + self.robot._right_joint_indices + action_dim,
                # right foot contact
                [6 + 2 * action_dim],
            )
        )

        # Do the same for left, except using +1 for left foot contact
        left_obs_indices = np.concatenate(
            (
                6 + self.robot._left_joint_indices,
                6 + self.robot._left_joint_indices + action_dim,
                [6 + 2 * action_dim + 1],
            )
        )

        negation_obs_indices = np.array(
            [
                2,  # vy
                4,  # roll
                6,  # abdomen_z pos
                8,  # abdomen_x pos
                27,  # abdomen_z vel
                29,  # abdomen_x vel
                50,  # sin(-a) = -sin(a) of next step
                53,  # x_tilt of next step
                55,  # sin(-a) = -sin(a) of next + 1 step
                58,  # x_tilt of next + 1 step
            ],
            dtype=np.int64,
        )

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        right_action_indices = self.robot._right_joint_indices
        left_action_indices = self.robot._left_joint_indices

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )
