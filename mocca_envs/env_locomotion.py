import math

import gym
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_objects import (
    VSphere,
    Pillar,
    Plank,
    LargePlank,
    HeightField,
    MonkeyBar,
)
from mocca_envs.robots import Child3D, Mike, Monkey3D, Walker3D


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
    robot_random_start = True
    robot_init_position = None

    def __init__(self, **kwargs):
        super().__init__(self.robot_class, **kwargs)
        self.robot.set_base_pose(pose="running_start")
        self.eval_mode = False

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

        # self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start, pos=self.robot_init_position
        )

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


class Walker3DStepperEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Walker3D
    # Pillar, Plank, LargePlank
    plank_class = LargePlank
    robot_random_start = True
    robot_init_position = [0.3, 0, 1.32]

    def __init__(self, **kwargs):

        # Need these before calling constructor
        # because they are used in self.create_terrain()
        self.step_radius = 0.25
        self.rendered_step_count = 4

        super().__init__(self.robot_class, remove_ground=True, **kwargs)
        self.robot.set_base_pose(pose="running_start")

        # Robot settings
        self.terminal_height = 0.8
        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.n_steps = 20
        self.lookahead = 2
        self.next_step_index = 0
        self.stop_frames_choices = [2.0]  # [15.0, 30.0]
        self.stop_frames = self.np_random.choice(self.stop_frames_choices)

        # Fix-ordered Curriculum
        self.curriculum = 0
        self.max_curriculum = 9

        # Terrain info
        self.dist_range = np.array([0.65, 1.05])
        self.pitch_range = np.array([-30, +30])  # degrees
        self.yaw_range = np.array([-20, 20])
        self.tilt_range = np.array([-10, 10])
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
        if self.rendered_step_count == self.n_steps:
            return

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

        # self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start, pos=self.robot_init_position
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

        info = {"steps_reached": self.next_step_index} if self.done else {}
        return state, reward, self.done, info

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

        self.tall_bonus = 2.0 if self.robot_state[0] > self.terminal_height else -1.0
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
                self.stop_frames = self.np_random.choice(self.stop_frames_choices)
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
            self.target_bonus = 4.0

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


class MikeStepperEnv(Walker3DStepperEnv):
    robot_class = Mike
    robot_init_position = (0.3, 0, 1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.is_rendered:
            self.robot.decorate()


class Monkey3DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    initial_height = 20
    bar_length = 5

    def __init__(self, render=False):
        # Need these before calling constructor
        # because they are used in self.create_terrain()
        self.step_radius = 0.015
        self.rendered_step_count = 4

        super().__init__(Monkey3D, render)
        self.robot.set_base_pose(pose="monkey_start")
        self.robot.base_position = (0, 0, self.initial_height)
        self.robot.base_velocity = np.array([3, 0, -1])

        # Robot settings
        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.n_steps = 32
        self.lookahead = 2
        self.next_step_index = 2

        # Terrain info
        self.pitch_limit = 0
        self.yaw_limit = 0
        self.r_range = np.array([0.3, 0.5])
        self.terrain_info = np.zeros((self.n_steps, 4))

        # robot_state + (2 targets) * (x, y, z) + {swing, pivot}_leg + swing_{xyz, quat}
        robot_obs_dim = self.robot.observation_space.shape[0]
        base_env_dim = robot_obs_dim + self.lookahead * 3 + 2 + 7
        high = np.inf * np.ones(base_env_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # torques
        robot_act_dim = self.robot.action_space.shape[0]
        high = np.inf * np.ones(robot_act_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.state_id = self._p.saveState()

    def generate_step_placements(self, n_steps=50, yaw_limit=30, pitch_limit=25):

        y_range = np.array([-yaw_limit, yaw_limit]) * DEG2RAD
        p_range = np.array([90 - pitch_limit, 90 + pitch_limit]) * DEG2RAD

        dr = self.np_random.uniform(*self.r_range, size=n_steps)
        dphi = self.np_random.uniform(*y_range, size=n_steps)
        dtheta = self.np_random.uniform(*p_range, size=n_steps)

        # special treatment for first steps
        dphi[0] = 0
        dphi[1] = 0

        phi = np.cumsum(dphi)

        dx = dr * np.sin(dtheta) * np.cos(phi + self.base_phi)
        # clip to prevent overlapping
        dx = np.sign(dx) * np.minimum(
            np.maximum(np.abs(dx), self.step_radius * 2.5), self.r_range[1]
        )
        dy = dr * np.sin(dtheta) * np.sin(phi + self.base_phi)
        dz = dr * np.cos(dtheta)

        # first step is on the arm that is behind
        i = np.argmin(self.robot.feet_xyz[:, 0])
        dx[0] = self.robot.feet_xyz[i, 0]
        dy[0] = self.robot.feet_xyz[i, 1]
        dz[0] = self.robot.feet_xyz[i, 2]

        # second step on the arm that is in front
        j = np.argmax(self.robot.feet_xyz[:, 0])
        dx[1] = self.robot.feet_xyz[j, 0] - dx[0] + 0.01
        dy[1] = self.robot.feet_xyz[j, 1] - dy[0]
        dz[1] = self.robot.feet_xyz[j, 2] - dz[0] - 0.02

        dx[0] += 0.04
        dz[0] += -self.initial_height + 0.04

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz) + self.initial_height

        self.swing_leg = i
        self.pivot_leg = j

        return np.stack((x, y, z, phi), axis=1)

    def create_terrain(self):
        self.steps = []
        step_ids = set()
        cover_ids = set()

        for index in range(self.rendered_step_count):
            p = MonkeyBar(self._p, self.step_radius, self.bar_length)
            self.steps.append(p)
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids) | self.ground_ids

    def set_step_state(self, info_index, step_index):
        pos = self.terrain_info[info_index, 0:3]
        phi = self.terrain_info[info_index, 3]
        quaternion = np.array(self._p.getQuaternionFromEuler([90 * DEG2RAD, 0, phi]))
        self.steps[step_index].set_position(pos=pos, quat=quaternion)

    def randomize_terrain(self):

        self.terrain_info = self.generate_step_placements(
            self.n_steps, self.yaw_limit, self.pitch_limit
        )

        for index in range(self.rendered_step_count):
            self.set_step_state(index, index)

    def update_steps(self):
        threshold = int(self.rendered_step_count // 2)
        if self.next_step_index >= threshold:
            oldest = (self.next_step_index - threshold - 1) % self.rendered_step_count

            next = min(
                (self.next_step_index - threshold - 1) + self.rendered_step_count,
                len(self.terrain_info) - 1,
            )
            self.set_step_state(next, oldest)

    def get_observation_component(self):
        swing_foot_name = "right_palm" if self.swing_leg == 0 else "left_palm"
        swing_foot_xyz = self.robot.parts[swing_foot_name].pose().xyz()
        swing_foot_delta = self.walk_target - swing_foot_xyz
        swing_foot_quat = self.robot.parts[swing_foot_name].pose().orientation()

        return (
            self.robot_state[:-2],
            self.robot.feet_contact,
            self.targets.flatten(),
            [self.swing_leg, self.pivot_leg],
            swing_foot_delta,
            swing_foot_quat,
        )

    def reset(self):
        self.done = False
        self.free_fall_count = 0
        self.target_reached_count = 0
        self.timestep = 0

        for i in range(self._p.getNumConstraints()):
            id = self._p.getConstraintUniqueId(i)
            self._p.removeConstraint(id)

        # start at 2 because first 2 are already in contact
        self.next_step_index = 2

        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(random_pose=False)

        self.base_phi = DEG2RAD * np.array(
            [-10] + [20, -20] * (self.n_steps // 2 - 1) + [10]
        )
        self.base_phi *= np.sign(float(not self.robot.mirrored) - 0.5)

        self.randomize_terrain()
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        # Order is important because walk_target is set up above
        self.calc_potential()

        state = np.concatenate(self.get_observation_component())

        return state

    def step(self, action):
        # action *= 0
        self.timestep += 1

        action[17 if self.swing_leg == 0 else 22] = +1
        action[17 if self.pivot_leg == 0 else 22] = -1

        # action[[17, 22]] = -1

        self.robot.apply_action(action)
        self.scene.global_step()

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state(contact_object_ids=None)
        self.calc_env_state(action)

        reward = self.progress - 0 * self.energy_penalty
        reward += self.step_bonus + (self.target_bonus - self.speed_penalty) * 0
        reward += 0 * (self.tall_bonus - self.posture_penalty - self.joints_penalty)

        contact_penalty = self.robot.feet_contact[self.swing_leg]
        reward += -contact_penalty

        self.done = self.done or (self.timestep > 180 and self.next_step_index <= 2)

        state = np.concatenate(self.get_observation_component())

        if self.is_rendered:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)

        return state, reward, self.done, {}

    def calc_potential(self):

        walk_target_delta = self.walk_target - self.robot.body_xyz
        self.distance_to_target = (
            walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2
        ) ** (1 / 2)

        swing_foot_name = "right_palm" if self.swing_leg == 0 else "left_palm"
        swing_foot_xyz = self.robot.parts[swing_foot_name].pose().xyz()
        swing_foot_delta = self.walk_target - swing_foot_xyz
        swing_distance = np.linalg.norm(swing_foot_delta)

        self.linear_potential = -self.distance_to_target / self.scene.dt
        self.swing_potential = -swing_distance / self.scene.dt

    def calc_base_reward(self, action):

        # Bookkeeping stuff
        old_linear_potential = self.linear_potential
        old_swing_potential = self.swing_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        swing_progress = self.swing_potential - old_swing_potential
        self.progress = 0 * linear_progress + swing_progress

        self.posture_penalty = 0
        if not -60 < self.robot.body_rpy[0] * RAD2DEG < 60:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        if not -40 < self.robot.body_rpy[1] * RAD2DEG < 40:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -90 < self.robot.body_rpy[2] * RAD2DEG < 90:
            self.posture_penalty += abs(self.robot.body_rpy[2])

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

        self.tall_bonus = 2.0 * float((self.robot.feet_contact == 1).any())
        self.done = self.done or (self.free_fall_count > 30)

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]
        p_xyz = self.terrain_info[self.next_step_index, [0, 1, 2]]

        for i, f in enumerate(self.robot.feet):

            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if self.all_contact_object_ids & contact_ids:
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

            if i == self.swing_leg:

                delta = self.robot.feet_xyz[self.swing_leg] - p_xyz
                distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
                self.foot_dist_to_target = distance

                palm_name = "right_palm" if self.swing_leg == 0 else "left_palm"
                palm_id = self.robot.parts[palm_name].bodyPartIndex
                palm_contacts = set(
                    (x[2], x[4])
                    for x in self._p.getContactPoints(self.robot.id, linkIndexA=palm_id)
                )

                # self.target_reached = bool(
                #     {(next_step.id, next_step.cover_id)} & contact_ids
                # ) and bool({(next_step.id, next_step.cover_id)} & finger_contacts)

                self.target_reached_count += bool(
                    {(next_step.id, next_step.cover_id)} & palm_contacts
                )

                self.target_reached = self.target_reached_count >= 1

                if not self.target_reached:
                    continue

                # Update next step
                self.target_reached_count = 0
                self.next_step_index += 1
                self.next_step_index = min(
                    self.next_step_index, self.terrain_info.shape[0] - 1
                )
                self.update_steps()
                self.pivot_leg = self.swing_leg
                self.swing_leg = (self.swing_leg + 1) % 2

    def calc_step_reward(self):

        self.step_bonus = 0
        if self.target_reached:
            self.step_bonus = 50 * np.exp(-self.foot_dist_to_target / 0.25)

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

        # use contact to detect done
        mask = float(np.sum(self.robot.feet_contact) == 0)
        self.free_fall_count = mask * self.free_fall_count + mask

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

        self.walk_target = targets[[0], 0:3].mean(axis=0)

        # delta_pos = targets[:, 0:3] - self.robot.feet_xyz[self.swing_leg]
        delta_pos = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.linalg.norm(delta_pos[:, 0:2], ord=2, axis=1)

        deltas = np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,  # x
                np.cos(angle_to_targets) * distance_to_targets,  # y
                delta_pos[:, 2],  # z
            ),
            axis=1,
        )

        return deltas

    def get_mirror_indices(self):

        global_dim = 6
        robot_act_dim = self.robot.action_space.shape[0]
        robot_obs_dim = self.robot.observation_space.shape[0]
        env_obs_dim = self.observation_space.shape[0]

        # _ + 6 accounting for global
        right = self.robot._right_joint_indices + global_dim
        # _ + robot_act_dim to get velocities, right foot contact
        right = np.concatenate(
            (right, right + robot_act_dim, [robot_obs_dim - 2, env_obs_dim - 2])
        )
        # Do the same for left
        left = self.robot._left_joint_indices + global_dim
        left = np.concatenate(
            (left, left + robot_act_dim, [robot_obs_dim - 1, env_obs_dim - 1])
        )

        # Used for creating mirrored observations
        # 2:  vy
        # 4:  roll
        # 6:  abdomen_z pos
        # 8:  abdomen_x pos
        # 27: abdomen_z vel
        # 29: abdomen_x vel
        # 54: sin(-a) = -sin(a) of next step
        # 57: sin(-a) = -sin(a) of next + 1 step
        negation_obs_indices = np.array(
            [
                2,
                4,
                6,
                8,
                6 + robot_act_dim,
                8 + robot_act_dim,
                robot_obs_dim,
                robot_obs_dim + 3,
            ],
            dtype=np.int64,
        )
        right_obs_indices = right
        left_obs_indices = left

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        # 23, 24 are for holding
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