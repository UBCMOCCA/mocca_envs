import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_utils import VSphere
from mocca_envs.robots import Walker3D

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

    def __init__(self, render=False):
        super(Walker3DCustomEnv, self).__init__(Walker3D, render)

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

    def reset(self):
        self.done = False
        self.add_angular_progress = True
        self.dist = self.np_random.uniform(0, 10)

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

        sin_ = self.distance_to_target * np.sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * np.cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))

        state = np.concatenate((self.robot_state, [sin_], [cos_]))

        return state

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        self.robot_state = self.robot.calc_state(self.ground_ids)
        self.calc_env_state(action)

        reward = self.progress + self.target_bonus - self.energy_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        sin_ = self.distance_to_target * np.sin(self.angle_to_target)
        sin_ = sin_ / (1 + abs(sin_))
        cos_ = self.distance_to_target * np.cos(self.angle_to_target)
        cos_ = cos_ / (1 + abs(cos_))

        state = np.concatenate((self.robot_state, [sin_], [cos_]))

        if self.is_render:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            if self.distance_to_target < 0.15:
                self.target.set_color(Colors["dodgerblue"])
            else:
                self.target.set_color(Colors["crimson"])

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

        if self.distance_to_target < 1:
            self.add_angular_progress = False

        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential

        self.progress = 2 * linear_progress
        if self.add_angular_progress:
            self.progress += 100 * angular_progress

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

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        # Order is important
        # calc_target_reward() potential
        self.calc_base_reward(action)
        self.calc_target_reward()

        if self.close_count >= self.np_random.choice([75, 150]):
            self.close_count = 0
            self.add_angular_progress = True
            angle = self.np_random.uniform(-np.pi, np.pi)
            delta = self.dist * np.array([np.cos(angle), np.sin(angle), 0.0])
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


class Walker3DTerrainEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    def __init__(self, render=False):

        # Need these before calling constructor
        # because they are used in self.create_terrain()
        self.step_radius = 0.25
        self.step_height = 0.1
        self.rendered_step_count = 4

        super(Walker3DTerrainEnv, self).__init__(Walker3D, render)

        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        # Terrain info
        self.n_steps = 24
        self.next_step_index = 0
        self.lookahead = 2
        # x, y, z, phi
        self.terrain_info = np.zeros((self.n_steps, 4))

        # (2 targets) * (x, y, z)
        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2 * 3)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def generate_step_placements(
        self, n_steps=50, min_gap=0.5, max_gap=0.85, phi_limit=30, theta_limit=25
    ):
        phi_limit = phi_limit * DEG2RAD
        theta_limit = theta_limit * DEG2RAD

        placements = np.zeros((n_steps, 4))

        # Set first two planks underneath feet
        placements[0, 0:3] = self.robot.feet_xyz[0]
        placements[1, 0:3] = self.robot.feet_xyz[1]
        placements[0:2, 2] -= 0.13  # Reset z to be flat on the ground

        step = 2

        while step < n_steps:

            dr = self.np_random.uniform(low=min_gap, high=max_gap)
            dphi = self.np_random.uniform(low=-phi_limit, high=phi_limit)
            dtheta = self.np_random.uniform(
                low=np.pi / 2 - theta_limit, high=np.pi / 2 + theta_limit
            )

            dphi += placements[step - 1, 3]
            x = dr * np.sin(dtheta) * np.cos(dphi)
            y = dr * np.sin(dtheta) * np.sin(dphi)
            z = dr * np.cos(dtheta)

            # Check for overlap

            start = max(0, step - self.rendered_step_count)
            prev_xyz = placements[start:step, 0:3]
            if prev_xyz.size > 0:
                proposal_xyz = placements[step - 1, 0:3] + (x, y, z)
                dist = np.linalg.norm((prev_xyz - proposal_xyz)[:, 0:2], ord=2, axis=1)
                overlapped = (dist < 2 * self.step_radius).any()

                if not overlapped:
                    placements[step, 0:3] = proposal_xyz
                    placements[step, 3] = dphi
                    step += 1

        np.clip(placements[:, 2], a_min=0.0, a_max=1.0, out=placements[:, 2])

        return placements

    def create_terrain(self):

        planks = []
        plank_ids = []
        covers = []
        cover_ids = []

        plank_shape = self._p.createCollisionShape(
            self._p.GEOM_CYLINDER, radius=self.step_radius, height=self.step_height * 2
        )
        cover_shape = self._p.createCollisionShape(
            self._p.GEOM_CYLINDER, radius=self.step_radius, height=0.01
        )
        plank_vshape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=self.step_radius,
            length=self.step_height * 2,
            rgbaColor=[88 / 255, 99 / 255, 110 / 255, 1],
            specularColor=[0.4, 0.4, 0],
        )
        cover_vshape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=self.step_radius,
            length=0.01,
            rgbaColor=[55 / 255, 55 / 255, 55 / 255, 1],
            specularColor=[0.4, 0.4, 0],
        )

        x, y, z = 1, 1, 1
        for index in range(self.rendered_step_count):
            oid = self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=plank_shape,
                baseVisualShapeIndex=plank_vshape,
                basePosition=[x, y, z - self.step_height],
            )
            self._p.changeDynamics(oid, -1, lateralFriction=1.0, restitution=0.1)
            planks.append(oid)
            plank_ids.append((oid, -1))

            oid = self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cover_shape,
                baseVisualShapeIndex=cover_vshape,
                basePosition=[x, y, z],
            )
            self._p.changeDynamics(oid, -1, lateralFriction=1.0, restitution=0.1)
            covers.append(oid)
            cover_ids.append((oid, -1))

        self.planks = planks
        self.covers = covers
        # Need set for detecting contact
        self.plank_ids = set(plank_ids)
        self.cover_ids = set(cover_ids)
        self.all_contact_object_ids = self.plank_ids | self.cover_ids | self.ground_ids

    def randomize_terrain(self):
        # Make flat terrain for now
        theta_limit = 0
        phi_limit = 180

        placements = self.generate_step_placements(
            n_steps=self.n_steps, theta_limit=theta_limit, phi_limit=phi_limit
        )

        for index, (x, y, z, phi) in enumerate(placements):
            self.terrain_info[index, :] = [x, y, z, phi]
            if index < self.rendered_step_count:
                pid = self.planks[index]
                cid = self.covers[index]
                quaternion = np.array(self._p.getQuaternionFromEuler([0, 0, phi]))
                self._p.resetBasePositionAndOrientation(
                    pid, posObj=(x, y, z - self.step_height), ornObj=quaternion
                )
                self._p.resetBasePositionAndOrientation(
                    cid, posObj=(x, y, z), ornObj=quaternion
                )

    def update_steps(self):
        if self.next_step_index > self.rendered_step_count - self.lookahead:
            i = (
                self.next_step_index + self.lookahead - self.rendered_step_count - 1
            ) % self.rendered_step_count
            pid = self.planks[i]
            cid = self.covers[i]

            if self.next_step_index + self.lookahead - 1 < len(self.terrain_info):
                x, y, z, phi = self.terrain_info[
                    self.next_step_index + self.lookahead - 1
                ]
            else:
                # At the end of the course, hopefully this doesn't happen
                x, y, z, phi = self.terrain_info[-1]

            quaternion = np.array(self._p.getQuaternionFromEuler([0, 0, phi]))
            self._p.resetBasePositionAndOrientation(
                pid, posObj=(x, y, z - self.step_height), ornObj=quaternion
            )
            self._p.resetBasePositionAndOrientation(
                cid, posObj=(x, y, z), ornObj=quaternion
            )

    def reset(self):
        self.done = False
        self.last_count = 0
        self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(random_pose=True, z0=2)
        self.calc_feet_state()

        # Randomize platforms
        self.randomize_terrain()
        self.next_step_index = 0

        # Reset camera
        if self.is_render:
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
        reward += self.step_bonus + self.target_bonus
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        state = np.concatenate((self.robot_state, self.targets.flatten()))

        if self.is_render:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self.target.set_position(pos=self.walk_target)
            if self.distance_to_target < 0.15:
                self.target.set_color(Colors["dodgerblue"])
            else:
                self.target.set_color(Colors["crimson"])

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
        self.progress = 2 * linear_progress

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

        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        self.tall_bonus = 1.5 if height > 0.7 else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_feet_state(self):
        # Calculate contact separately for planks
        target_cover_index = self.next_step_index % self.rendered_step_count
        target_cover_id = {(self.covers[target_cover_index], -1)}

        target_reached = False
        foot_dist_to_next_step = [0.0, 0.0]

        centre_distance = float("inf")
        p_xyz = self.terrain_info[self.next_step_index, [0, 1, 2]]
        for i, f in enumerate(self.robot.feet):
            self.robot.feet_xyz[i] = f.pose().xyz()
            contact_ids = set((x[2], x[4]) for x in f.contact_list())

            in_contact = self.all_contact_object_ids & contact_ids
            self.robot.feet_contact[i] = 1.0 if in_contact else 0.0

            delta = self.robot.feet_xyz[i] - p_xyz
            distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
            foot_dist_to_next_step[i] = distance

            if not target_cover_id & contact_ids and self.ground_ids & contact_ids:
                self.done = True

            if target_cover_id & contact_ids:
                target_reached = True
                if distance < centre_distance:
                    centre_distance = distance

        return target_reached, centre_distance, foot_dist_to_next_step

    def calc_terrain_reward(self):

        self.target_reached, centre_distance, foot_dist_to_next_step = (
            self.calc_feet_state()
        )

        self.step_bonus = 0
        if self.target_reached:
            self.step_bonus = 50 * 2.718 ** (-centre_distance / 0.25)

            self.next_step_index += 1
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1
                self.last_count += 1

            if self.last_count > 1:
                self.step_bonus = 0

            self.update_steps()

        self.target_bonus = 0
        if self.distance_to_target < 0.15:
            # Mostly for last step only
            self.target_bonus = 2.0

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        self.calc_base_reward(action)
        # detects contact and set next step
        self.calc_terrain_reward()
        # use next step to calculate next k steps
        self.targets = self.delta_to_k_targets(k=self.lookahead)
        # Order is important because walk_target is set up above
        self.calc_potential()

    def delta_to_k_targets(self, k=1):
        """ Return positions (relative to root) of target, and k-1 planks after """
        targets = self.terrain_info[
            self.next_step_index : self.next_step_index + k, [0, 1, 2]
        ]
        if len(targets) < k:
            # If running out of targets, repeat last target
            targets = np.concatenate(
                (targets, np.repeat(targets[[-1]], k - len(targets), axis=0))
            )

        self.walk_target = targets[:, 0:3].mean(axis=0)

        deltas = targets - self.robot.body_xyz
        target_thetas = np.arctan2(deltas[:, 1], deltas[:, 0])

        self.angle_to_targets = target_thetas - self.robot.body_rpy[2]
        self.distance_to_targets = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)

        deltas = np.stack(
            (
                np.sin(self.angle_to_targets) * self.distance_to_targets,
                np.cos(self.angle_to_targets) * self.distance_to_targets,
                deltas[:, 2],
            ),
            axis=1,
        )

        # Normalize targets to between -1 and +1 using softsign
        deltas /= 1 + np.abs(deltas)

        return deltas

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
        # 50: sin(-a) = -sin(a) of next step
        # 53: sin(-a) = -sin(a) of next + 1 step
        negation_obs_indices = np.array([2, 4, 6, 8, 27, 29, 50, 53], dtype=np.int64)
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
