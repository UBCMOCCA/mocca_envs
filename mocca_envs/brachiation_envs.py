import gym
import numpy as np

from mocca_envs.bullet_objects import MonkeyBar
from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Monkey3D


DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class Monkey3DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 8

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

        # robot_state + (2 targets) * (x, y, z) + {swing, pivot}_leg
        robot_obs_dim = self.robot.observation_space.shape[0]
        high = np.inf * np.ones(robot_obs_dim + self.lookahead * 3 + 2)
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
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

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

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

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
        return (
            self.robot_state[:-2],
            self.robot.feet_contact,
            self.targets.flatten(),
            [self.swing_leg, self.pivot_leg],
        )

    def reset(self):
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

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

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

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
        reward += 0 * self.tall_bonus - self.posture_penalty - self.joints_penalty

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

        swing_foot_xyz = self.robot.feet_xyz[self.swing_leg]
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
        self.progress = linear_progress + swing_progress

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
