from math import sin, cos, atan2, sqrt

import gym
import numpy as np
import pybullet

from numba import njit
from numpy import concatenate
from bottleneck import ss, anynan, nanargmax, nanargmin, nanmin

from mocca_envs.bullet_objects import VSphere, Pillar, Plank, LargePlank
from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Mike, Walker3D

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class Walker3DAgileEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4
    max_timestep = 1000

    robot_class = Walker3D
    robot_random_start = True
    robot_init_position = [0, 0, 1.32]
    robot_init_velocity = None

    plank_class = LargePlank  # Pillar, Plank, LargePlank
    num_steps = 20
    step_radius = 0.25
    rendered_step_count = 3
    init_step_separation = 0.75

    lookahead = 2
    lookbehind = 1
    walk_target_index = -1
    step_bonus_smoothness = 1
    stop_steps = [6, 7, 13, 14]

    def __init__(self, **kwargs):
        # Handle non-robot kwargs
        plank_name = kwargs.pop("plank_class", None)
        self.plank_class = globals().get(plank_name, self.plank_class)

        super().__init__(self.robot_class, remove_ground=True, **kwargs)
        # self.robot.set_base_pose(pose="running_start")
        # self.robot_init_position = [0.3, 0, 1.32]

        # Fix-ordered Curriculum
        self.curriculum = 0
        self.max_curriculum = 9
        self.advance_threshold = 12  # steps_reached

        # Robot settings
        N = self.max_curriculum + 1
        self.terminal_height_curriculum = np.linspace(0.75, 0.45, N)
        self.applied_gain_curriculum = np.linspace(1.0, 1.2, N)
        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.next_step_index = self.lookbehind

        # Terrain info
        self.dist_range = np.array([0.65, 1.25])
        self.pitch_range = np.array([0, 0])  # degrees
        self.yaw_range = np.array([-20, 20])
        self.tilt_range = np.array([0, 0])
        self.step_param_dim = 5
        # Important to do this once before reset!
        self.terrain_info = self.generate_step_placements()

        # Observation and Action spaces
        self.robot_obs_dim = self.robot.observation_space.shape[0]
        K = self.lookahead + self.lookbehind
        high = np.inf * np.ones(self.robot_obs_dim + K * self.step_param_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

        # pre-allocate buffers
        F = len(self.robot.feet)
        self._foot_target_contacts = np.zeros((F, 1), dtype=np.float32)
        self.foot_dist_to_target = np.zeros(F, dtype=np.float32)

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

        N = self.num_steps
        rng = self.np_random

        dr = rng.uniform(*dist_range, size=N)
        dphi = rng.uniform(*yaw_range, size=N)
        dtheta = rng.uniform(*pitch_range, size=N)
        x_tilt = rng.uniform(*tilt_range, size=N)
        y_tilt = rng.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1:3] = self.init_step_separation
        dphi[1:3] = 0.0
        dtheta[1:3] = np.pi / 2

        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        phi0 = rng.choice(np.linspace(0, 2 * np.pi, 20))
        phi = np.cumsum(dphi) + phi0

        dx = dr * np.sin(dtheta) * np.cos(phi)
        dy = dr * np.sin(dtheta) * np.sin(phi)
        dz = dr * np.cos(dtheta)

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        return np.stack((x, y, z, phi, x_tilt, y_tilt), axis=1)

    def create_terrain(self):

        self.steps = []
        step_ids = set()
        cover_ids = set()

        options = {
            # self._p.URDF_ENABLE_SLEEPING |
            "flags": self._p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        }

        for index in range(self.rendered_step_count):
            p = self.plank_class(self._p, self.step_radius, options=options)
            self.steps.append(p)
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids)

        if not self.remove_ground:
            self.all_contact_object_ids |= self.ground_ids

    def set_step_state(self, info_index, step_index):
        pos = self.terrain_info[info_index, 0:3]
        phi, x_tilt, y_tilt = self.terrain_info[info_index, 3:6]
        quaternion = np.array(pybullet.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
        self.steps[step_index].set_position(pos=pos, quat=quaternion)

    def randomize_terrain(self, replace=True):
        if replace:
            self.terrain_info = self.generate_step_placements()
        for index in range(self.rendered_step_count):
            self.set_step_state(index, index)

    def update_steps(self):
        if self.rendered_step_count == self.num_steps:
            return

        if self.next_step_index >= self.rendered_step_count:
            oldest = self.next_step_index % self.rendered_step_count
            next = min(self.next_step_index, len(self.terrain_info) - 1)
            self.set_step_state(next, oldest)

    def reset(self):
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        self.timestep = 0
        self.done = False
        self.target_reached_count = 0

        self.set_stop_on_next_step = False
        self.stop_on_next_step = False

        self.robot.applied_gain = self.applied_gain_curriculum[self.curriculum]
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
        )
        self.swing_leg = 1 if self.robot.mirrored else 0

        # Randomize platforms
        replace = self.next_step_index >= self.num_steps / 2
        self.next_step_index = self.lookbehind
        self._prev_next_step_index = self.next_step_index - 1
        self.randomize_terrain(replace)
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets()
        assert self.targets.shape[-1] == self.step_param_dim

        # Order is important because walk_target is set up above
        self.calc_potential()

        state = concatenate((self.robot_state, self.targets.flatten()))

        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        return state

    def step(self, action):
        self.timestep += 1

        self.robot.apply_action(action)
        self.scene.global_step()

        # Stop on the 7th and 14th step, but need to specify N-1 as well
        self.set_stop_on_next_step = self.next_step_index in self.stop_steps

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)

        reward = self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        # targets is calculated by calc_env_state()
        state = concatenate((self.robot_state, self.targets.flatten()))

        if self.is_rendered or self.use_egl:
            self._handle_keyboard(callback=self.handle_keyboard)
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)

        info = {}
        if self.done or self.timestep == self.max_timestep - 1:
            info["curriculum_metric"] = self.next_step_index

        return state, reward, self.done, info

    def handle_keyboard(self, keys):
        RELEASED = self._p.KEY_WAS_RELEASED

        # stop at current
        if keys.get(ord("s")) == RELEASED:
            self.set_stop_on_next_step = not self.set_stop_on_next_step

    def create_target(self):
        # Need this to create target in render mode, called by EnvBase
        # Sphere is a visual shape, does not interact physically
        self.target = VSphere(self._p, radius=0.15, pos=None)

    def calc_potential(self):
        walk_target_delta = self.walk_target - self.robot.body_xyz
        self.distance_to_target = sqrt(ss(walk_target_delta[0:2]))
        self.linear_potential = -self.distance_to_target / self.scene.dt

    @staticmethod
    @njit(fastmath=True)
    def _calc_base_reward(action, joint_speed, electricity_coef, stall_torque_coef):
        electricity_sum = 0
        stall_torque_sum = 0
        N = len(action)

        for i in range(N):
            electricity_sum += abs(action[i] * joint_speed[i])
            stall_torque_sum += action[i] ** 2

        electricity_cost = electricity_coef * electricity_sum / N
        stall_torque_cost = stall_torque_coef * stall_torque_sum / N
        energy_penalty = electricity_cost + stall_torque_cost

        return energy_penalty

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

        # speed = sqrt(ss(self.robot.body_vel))
        # self.speed_penalty = max(speed - 1.6, 0)

        self.energy_penalty = self._calc_base_reward(
            action,
            self.robot.joint_speeds,
            self.electricity_cost,
            self.stall_torque_cost,
        )

        self.joints_penalty = self.joints_at_limit_cost * self.robot.joints_at_limit

        terminal_height = self.terminal_height_curriculum[self.curriculum]
        self.tall_bonus = 2.0 if self.robot_state[0] > terminal_height else -1.0
        abs_height = self.robot.body_xyz[2] - self.terrain_info[self.next_step_index, 2]
        strafing = -0.4 < self.robot.body_rpy[2] < 0.4
        self.done = self.done or self.tall_bonus < 0 or abs_height < -3 or not strafing

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]
        # target_cover_id = {(next_step.id, next_step.cover_id)}

        self.foot_dist_to_target = np.sqrt(
            ss(
                self.robot.feet_xyz[:, 0:2]
                - self.terrain_info[self.next_step_index, 0:2],
                axis=1,
            )
        )

        robot_id = self.robot.id
        client_id = self._p._client
        target_id_list = [next_step.id]
        target_cover_id_list = [next_step.cover_id]
        self._foot_target_contacts.fill(0)

        for i, (foot, contact) in enumerate(
            zip(self.robot.feet, self._foot_target_contacts)
        ):
            self.robot.feet_contact[i] = pybullet.getContactStates(
                bodyA=robot_id,
                linkIndexA=foot.bodyPartIndex,
                bodiesB=target_id_list,
                linkIndicesB=target_cover_id_list,
                results=contact,
                physicsClientId=client_id,
            )

        if (
            self.next_step_index - 1 in self.stop_steps
            and self.next_step_index - 2 in self.stop_steps
        ):
            self.swing_leg = nanargmax(self._foot_target_contacts[:, 0])
        self.target_reached = self._foot_target_contacts[self.swing_leg, 0] > 0

        # At least one foot is on the plank
        if self.target_reached:
            self.target_reached_count += 1

            # Advance after has stopped for awhile
            if self.target_reached_count > 120:
                self.stop_on_next_step = False
                self.set_stop_on_next_step = False

            # Slight delay for target advancement
            # Needed for not over counting step bonus
            if self.target_reached_count >= 2:
                if not self.stop_on_next_step:
                    self.swing_leg = (self.swing_leg + 1) % 2
                    self.next_step_index += 1
                    self.target_reached_count = 0
                    self.update_steps()
                self.stop_on_next_step = self.set_stop_on_next_step

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
            dist = nanmin(self.foot_dist_to_target)
            self.step_bonus = 50 * 2.718 ** (
                -(dist ** self.step_bonus_smoothness) / 0.25
            )

        # For remaining stationary
        self.target_bonus = 0
        last_step = self.next_step_index == len(self.terrain_info) - 1
        if (last_step or self.stop_on_next_step) and self.distance_to_target < 0.15:
            self.target_bonus = 2.0

    def calc_env_state(self, action):
        if anynan(self.robot_state):
            print("~INF~", self.robot_state)
            self.done = True

        cur_step_index = self.next_step_index

        # detects contact and set next step
        self.calc_feet_state()
        self.calc_base_reward(action)
        self.calc_step_reward()
        # use next step to calculate next k steps
        self.targets = self.delta_to_k_targets()

        if cur_step_index != self.next_step_index:
            self.calc_potential()

    def delta_to_k_targets(self):
        """ Return positions (relative to root) of target, and k-1 step after """
        k = self.lookahead
        j = self.lookbehind
        N = self.next_step_index
        if self._prev_next_step_index != self.next_step_index:
            if not self.stop_on_next_step:
                if N - j >= 0:
                    targets = self.terrain_info[N - j : N + k]
                else:
                    targets = concatenate(
                        (
                            [self.terrain_info[0]] * j,
                            self.terrain_info[N : N + k],
                        )
                    )
                if len(targets) < (k + j):
                    # If running out of targets, repeat last target
                    targets = concatenate(
                        (targets, [targets[-1]] * ((k + j) - len(targets)))
                    )
            else:
                targets = concatenate(
                    (
                        self.terrain_info[N - j : N],
                        [self.terrain_info[N]] * k,
                    )
                )
            self._prev_next_step_index = self.next_step_index
            self._targets = targets
        else:
            targets = self._targets

        self.walk_target = targets[self.walk_target_index, 0:3]

        delta_pos = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.sqrt(ss(delta_pos[:, 0:2], axis=1))

        deltas = concatenate(
            (
                (np.sin(angle_to_targets) * distance_to_targets)[:, None],  # x
                (np.cos(angle_to_targets) * distance_to_targets)[:, None],  # y
                (delta_pos[:, 2])[:, None],  # z
                (targets[:, 4])[:, None],  # x_tilt
                (targets[:, 5])[:, None],  # y_tilt
            ),
            axis=1,
        )

        return deltas

    def get_mirror_indices(self):

        action_dim = self.robot.action_space.shape[0]

        right_obs_indices = concatenate(
            (
                # joint angle indices + 6 accounting for global
                6 + self.robot._right_joint_indices,
                # joint velocity indices
                6 + self.robot._right_joint_indices + action_dim,
                # right foot contact
                [
                    6 + 2 * action_dim + 2 * i
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )

        # Do the same for left, except using +1 for left foot contact
        left_obs_indices = concatenate(
            (
                6 + self.robot._left_joint_indices,
                6 + self.robot._left_joint_indices + action_dim,
                [
                    6 + 2 * action_dim + 2 * i + 1
                    for i in range(len(self.robot.foot_names) // 2)
                ],
            )
        )

        robot_neg_obs_indices = concatenate(
            (
                # vy, roll
                [2, 4],
                # negate part of robot (position)
                6 + self.robot._negation_joint_indices,
                # negate part of robot (velocity)
                6 + self.robot._negation_joint_indices + action_dim,
            )
        )

        steps_neg_obs_indices = np.array(
            [
                (
                    i * self.step_param_dim + 0,  # sin(-x) = -sin(x)
                    i * self.step_param_dim + 3,  # x_tilt
                )
                for i in range(self.lookahead + self.lookbehind)
            ],
            dtype=np.int64,
        ).flatten()

        negation_obs_indices = concatenate(
            (robot_neg_obs_indices, steps_neg_obs_indices + self.robot_obs_dim)
        )

        # Used for creating mirrored actions
        negation_action_indices = self.robot._negation_joint_indices
        right_action_indices = self.robot._right_joint_indices
        left_action_indices = self.robot._left_joint_indices

        obs_dim = self.observation_space.shape[0]
        assert len(negation_obs_indices) == 0 or negation_obs_indices.max() < obs_dim
        assert right_obs_indices.max() < obs_dim
        assert left_obs_indices.max() < obs_dim
        assert (
            len(negation_action_indices) == 0
            or negation_action_indices.max() < action_dim
        )
        assert right_action_indices.max() < action_dim
        assert left_action_indices.max() < action_dim

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )


class MikeAgileEnv(Walker3DAgileEnv):
    robot_class = Mike
    robot_init_position = [0, 0, 1.0]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.is_rendered:
            self.robot.decorate()
