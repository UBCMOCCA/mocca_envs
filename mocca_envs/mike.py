import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import gym
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_objects import VSphere, Pillar, Plank, LargePlank
from mocca_envs.robots import Walker2D, Walker3D, Mike
from mocca_envs.bullet_objects import HeightField

import torch
from mocca_envs.model import ActorCriticNet

Colors = {
    "dodgerblue": (0.11764705882352941, 0.5647058823529412, 1.0, 1.0),
    "crimson": (0.8627450980392157, 0.0784313725490196, 0.23529411764705882, 1.0),
}

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class MikeStepperEnv(EnvBase):
    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    def __init__(self, render=False):

        self.make_phantoms_yes = False
        self.phantoms = []
        self.current_phantom_idx = 0
        self.done_height = 0.7

        # Need these before calling constructor
        # because they are used in self.create_terrain()
        self.step_radius = 0.25
        self.rendered_step_count = 30
        self.stop_frames = 30

        super().__init__(Mike, render)
        self.robot.set_base_pose(pose="running_start")

        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        self.n_steps = 40
        self.lookahead = 2
        self.next_step_index = 0

        # self.base_phi = DEG2RAD * np.array(
        #     [-10] + [20, -20] * (self.n_steps // 2 - 1) + [10]
        # )

        # Terrain info
        self.pitch_limit = 20
        self.max_curriculum = 5
        self.yaw_limit = 10
        self.tilt_limit = 0
        self.r_range = np.array([0.65, 1.5])

        self.sample_size = 11
        self.yaw_sample_size = 11
        self.pitch_sample_size = 11
        self.r_sample_size = 11
        self.x_tilt_sample_size = 11
        self.y_tilt_sample_size = 11

        self.yaw_samples = np.linspace(-20, 20, num=self.yaw_sample_size) * DEG2RAD
        self.pitch_samples = np.linspace(-50, 50, num=self.pitch_sample_size) * DEG2RAD
        self.r_samples = np.linspace(0.65, 0.65, num=self.r_sample_size)
        self.x_tilt_samples = np.linspace(0, 0, num=self.x_tilt_sample_size) * DEG2RAD
        self.y_tilt_samples = np.linspace(0, 0, num=self.y_tilt_sample_size) * DEG2RAD

        self.yaw_pitch_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size)) / (
                self.yaw_sample_size * self.pitch_sample_size)
        self.yaw_pitch_r_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size)) / (
                self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size)
        self.yaw_pitch_r_tilt_prob = np.ones((self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size,
                                              self.x_tilt_sample_size, self.y_tilt_sample_size)) / (
                                             self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size * self.x_tilt_sample_size * self.y_tilt_sample_size)

        self.fake_yaw_samples = np.linspace(-20, 20, num=self.yaw_sample_size) * DEG2RAD
        self.fake_pitch_samples = np.linspace(-50, 50, num=self.pitch_sample_size) * DEG2RAD
        self.fake_r_samples = np.linspace(0.65, 1.5, num=self.r_sample_size)
        self.fake_x_tilt_samples = np.linspace(-20, 20, num=self.x_tilt_sample_size) * DEG2RAD
        self.fake_y_tilt_samples = np.linspace(-20, 20, num=self.y_tilt_sample_size) * DEG2RAD

        # x, y, z, phi, x_tilt, y_tilt
        self.terrain_info = np.zeros((self.n_steps, 6))

        # (2 targets) * (x, y, z, x_tilt, y_tilt)
        high = np.inf * np.ones(
            self.robot.observation_space.shape[0] + self.lookahead * 5
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space
        self.temp_states = np.zeros((self.sample_size ** 2, self.observation_space.shape[0]))
        # print(self.observation_space.shape)

        # add lots of phantoms

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        if self.make_phantoms_yes:
            for _ in range(20):
                phantom = self.robot_class(self._p, **self.robot_kwargs)
                phantom.np_random = self.np_random
                phantom.initialize()
                # set the phantom pose to current pose
                self.phantoms.append(phantom)

                # add the phantom to the environment and set collision filter
                self.scene.actor_introduce(phantom)
                for _, body_part in phantom.parts.items():
                    self._p.setCollisionFilterGroupMask(phantom.object_id[0], -1, 0, 0)
                    self._p.setCollisionFilterGroupMask(phantom.object_id[0], body_part.bodyPartIndex, 0, 0)
                    self._p.changeDynamics(phantom.object_id[0], -1, mass=0)
                    self._p.changeDynamics(phantom.object_id[0], body_part.bodyPartIndex, mass=0)

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def set_mirror(self, mirror):
        pass

    def update_specialist_2(self, specialist):
        self.specialist = min(specialist, 5)
        prev_specialist = self.specialist - 1
        # print((self.specialist * 2 + 1)**2 - (prev_specialist*2+1)**2)
        half_size = (self.sample_size - 1) // 2
        if specialist == 0:
            prob = 1
        else:
            prob = 1.0 / ((self.specialist * 2 + 1) ** 2 - (prev_specialist * 2 + 1) ** 2)
        window = slice(half_size - self.specialist, half_size + self.specialist + 1)
        prev_window = slice(half_size - prev_specialist, half_size + prev_specialist + 1)
        self.yaw_pitch_prob *= 0
        self.yaw_pitch_prob[window, window] = prob
        self.yaw_pitch_prob[prev_window, prev_window] = 0
        print(np.round(self.yaw_pitch_prob, 2))

    def update_specialist(self, specialist):
        self.specialist = min(specialist, 5)
        prev_specialist = self.specialist - 1
        # print((self.specialist * 2 + 1)**2 - (prev_specialist*2+1)**2)
        half_size = (self.sample_size - 1) // 2
        if specialist == 0:
            prob = 1
        else:
            prob = 1.0 / ((self.specialist * 2 + 1) ** 4 - (prev_specialist * 2 + 1) ** 4) / 3
            # prob = 1/4
        window = slice(half_size - self.specialist, half_size + self.specialist + 1)
        prev_window = slice(half_size - prev_specialist, half_size + prev_specialist + 1)
        r_index = [0]
        if self.specialist > 0:
            r_index.append(specialist * 2 - 1)
            r_index.append(specialist * 2)
        self.yaw_pitch_r_tilt_prob *= 0
        self.yaw_pitch_r_tilt_prob[window, window, r_index, window, window] = prob
        self.yaw_pitch_r_tilt_prob[prev_window, prev_window, r_index, prev_window, prev_window] = 0

    def generate_step_placements(
            self,
            n_steps=50,
            yaw_limit=30,
            pitch_limit=25,
            tilt_limit=10,
    ):

        y_range = np.array([-10, -10]) * DEG2RAD
        p_range = np.array([90 - 0, 90 + 0]) * DEG2RAD
        t_range = np.array([-20, -20]) * DEG2RAD

        dr = self.np_random.uniform(0.65, 0.65, size=n_steps)
        dphi = self.np_random.uniform(*y_range, size=n_steps)
        dtheta = self.np_random.uniform(*p_range, size=n_steps)
        # dtheta = np.array([75, 70] * 12) * DEG2RAD

        # dr = self.np_random.randint(0, 11, n_steps) / 10.0
        # dr = dr * min_gap + (1-dr) * max_gap
        # dphi = self.np_random.randint(0, 11, n_steps) / 10.0
        # dphi = dphi * y_range[0] + (1-dphi)*y_range[1]
        # dtheta = self.np_random.randint(0, 11, n_steps) / 10.0
        # dtheta = dtheta * p_range[0] + (1-dtheta)*p_range[1]

        # dr[0] = 0.8
        # dphi[0] = 0.0
        # dtheta[0] = np.pi / 2

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        # make first step slightly further to accommodate different starting poses
        dr[1] = 0.8
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dr[2] = 0.75
        dphi[2] = 0.0
        dtheta[2] = np.pi / 2

        # test_r = 0.75
        # dr[3] = test_r
        # dtheta[3] = (90 + 50) * DEG2RAD
        # for i in range(10):
        #     dr[3 + i * 2] = test_r
        #     dphi[3 + i * 2] = 0 * DEG2RAD
        #     dtheta[3 + i * 2] = (90 + 20) * DEG2RAD

        #     dr[4 + i * 2] = test_r
        #     dphi[4 + i * 2] = 0 * DEG2RAD
        #     dtheta[4 + i * 2] = (90 + 20) * DEG2RAD

        x_tilt = self.np_random.uniform(*t_range, size=n_steps)
        y_tilt = self.np_random.uniform(0, 0, size=n_steps)
        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi_copy = np.copy(dphi)
        dphi = np.cumsum(dphi)

        x_ = dr * np.sin(dtheta) * np.cos(dphi + self.base_phi)
        x_[2:] = np.sign(x_[2:]) * np.minimum(np.maximum(np.abs(x_[2:]), self.step_radius * 2.5), self.r_range[1])
        y_ = dr * np.sin(dtheta) * np.sin(dphi + self.base_phi)
        z_ = dr * np.cos(dtheta)

        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + 0

        min_z = self.step_radius * np.sin(self.tilt_limit * DEG2RAD) + 0.01
        np.clip(z, a_min=min_z, a_max=None, out=z)

        terrain_info = np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

        for i in range(2, self.n_steps - 1):
            next_step_xyz = terrain_info[i]
            bound_checked_index = (i + 1) % self.n_steps

            base_phi = self.base_phi[bound_checked_index]
            base_yaw = terrain_info[i, 3]

            pitch = dtheta[bound_checked_index]
            yaw = dphi_copy[bound_checked_index]

            dx = dr[bound_checked_index] * np.sin(pitch) * np.cos(yaw + base_phi)
            # clip to prevent overlapping
            dx = np.sign(dx) * min(max(abs(dx), self.step_radius * 2.5), self.r_range[1])
            dy = dr[bound_checked_index] * np.sin(pitch) * np.sin(yaw + base_phi)

            matrix = np.array([
                [np.cos(base_yaw), -np.sin(base_yaw)],
                [np.sin(base_yaw), np.cos(base_yaw)]
            ])

            dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

            x = next_step_xyz[0] + dxy[0]
            y = next_step_xyz[1] + dxy[1]
            z = next_step_xyz[2] + dr[bound_checked_index] * np.cos(pitch)

            terrain_info[bound_checked_index, 0] = x
            terrain_info[bound_checked_index, 1] = y
            terrain_info[bound_checked_index, 2] = z
            terrain_info[bound_checked_index, 3] = yaw + base_yaw
            terrain_info[bound_checked_index, 4] = -20 * DEG2RAD
            # terrain_info[bound_checked_index, 5] = 0#y_tilt

        return terrain_info

    def create_terrain(self):
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.steps = []
        step_ids = set()
        cover_ids = set()

        for index in range(self.rendered_step_count):
            # p = Pillar(self._p, self.step_radius)
            p = Plank(self._p, self.step_radius)
            # p = LargePlank(self._p, self.step_radius)
            self.steps.append(p)
            step_ids = step_ids | {(p.id, p.base_id)}
            cover_ids = cover_ids | {(p.id, p.cover_id)}

        # Need set for detecting contact
        self.all_contact_object_ids = set(step_ids) | set(cover_ids) | self.ground_ids
        from mocca_envs.bullet_objects import HeightField
        self.terrain = HeightField(self._p, (256, 256))
        self.ground_ids = {(self.terrain, -1)}

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def randomize_terrain(self):

        self.terrain_info = self.generate_step_placements(
            n_steps=self.n_steps,
            pitch_limit=self.pitch_limit,
            yaw_limit=self.yaw_limit,
            tilt_limit=self.tilt_limit,
        )

        for index in range(self.rendered_step_count):
            pos = np.copy(self.terrain_info[index, 0:3])
            #pos[2] -= 20
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
            pos = np.copy(self.terrain_info[next, 0:3])
            phi, x_tilt, y_tilt = self.terrain_info[next, 3:6]
            quaternion = np.array(self._p.getQuaternionFromEuler([x_tilt, y_tilt, phi]))
            self.steps[oldest].set_position(pos=pos, quat=quaternion)

    def reset(self):
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.robot.set_base_pose(pose="running_start")
        self.done = False
        self.target_reached_count = 0

        # self._p.restoreState(self.state_id)

        # self.robot_state = self.robot.reset(random_pose=True)
        self.robot_state = self.robot.reset(random_pose=False, pos=(0.3, 0, 1.), vel=[0.0, 0, 0])
        self.base_phi = DEG2RAD * np.array(
            [-10] + [20, -20] * (self.n_steps // 2 - 1) + [10]
        )
        self.base_phi *= np.sign(float(self.robot.mirrored) - 0.5)
        self.calc_feet_state()

        # Randomize platforms
        self.randomize_terrain()
        # self.terrain.generate_height_field_from_step_2d(self.terrain_info)
        #self.terrain_info = self.terrain.get_p_noise_height_field()
        self.next_step_index = 1

        self.next_next_pitch = np.pi / 2
        self.next_next_yaw = 0
        self.next_x_tilt = 0
        self.next_y_tilt = 0
        self.next_next_x_tilt = 0
        self.next_next_y_tilt = 0
        self.next_dr = 0.8
        self.next_next_dr = 0.8

        # Reset camera
        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        # Order is important because walk_target is set up above
        self.calc_potential()

        state = np.concatenate((self.robot_state, self.targets.flatten()))
        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        state[0] = height
        # import time; time.sleep(5)

        self.current_phantom_idx = 0
        if self.make_phantoms_yes:
            self.current_phantom_idx = 0
            for phantom in self.phantoms:
                phantom.reset(pos=[-1000, -1000, -1000])

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        return state

    def get_state(self):
        state = self.robot.calc_state()
        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        state[0] = height
        return state

    def step(self, action):

        self.robot.apply_action(action)
        self.scene.global_step()

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)

        reward = 1.0 * self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus - self.speed_penalty
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

        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        state[0] = height
        # print(height)

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
        self.speed_penalty = max(speed - 1.6, 0) * 0
        # print(speed)

        self.energy_penalty = self.electricity_cost * float(
            np.abs(action * self.robot.joint_speeds).mean()
        )
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())

        self.joints_penalty = float(
            self.joints_at_limit_cost * self.robot.joints_at_limit
        )

        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        self.tall_bonus = 2.0 if height > self.done_height else -1.0
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
            self.robot.feet_xyz[i] = f.pose().xyz()
            contact_ids = set((x[2], x[4]) for x in f.contact_list())

            # if contact_ids is not empty, then foot is in contact
            self.robot.feet_contact[i] = 1.0 if contact_ids else 0.0

            delta = self.robot.feet_xyz[i] - p_xyz
            distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
            self.foot_dist_to_target[i] = distance

            # if target_cover_id & contact_ids:
            #     self.target_reached = True
            if contact_ids and ((delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) < 0.3):
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
        if self.target_reached and self.target_reached_count == 1:
            self.step_bonus = 50 * np.exp(-self.foot_dist_to_target.min() / 0.25)

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

        self.update_terrain = (cur_step_index != self.next_step_index)

        if cur_step_index != self.next_step_index:
            self.update_terrain_info()
            self.calc_potential()

    def delta_to_k_targets(self, k=1):
        """ Return positions (relative to root) of target, and k-1 step after """
        targets = self.terrain_info[self.next_step_index: self.next_step_index + k]
        if len(targets) < k:
            # If running out of targets, repeat last target
            targets = np.concatenate(
                (targets, np.repeat(targets[[-1]], k - len(targets), axis=0))
            )

        self.walk_target = targets[[1], 0:3].mean(axis=0)

        deltas = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(deltas[:, 1], deltas[:, 0])

        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)

        deltas = np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,  # x
                np.cos(angle_to_targets) * distance_to_targets,  # y
                deltas[:, 2],  # z
                targets[:, 4],  # x_tilt
                targets[:, 5],  # y_tilt
            ),
            axis=1,
        )

        # Normalize targets x,y to between -1 and +1 using softsign
        # deltas[:, 0:2] /= 1 + np.abs(deltas[:, 0:2])

        return deltas

    def update_terrain_info(self):
        # print(env.next_step_index)
        next_next_step = self.next_step_index + 1
        # env.terrain_info[next_next_step, 2] = 30    
        self.sample_next_next_step()
        # +1 because first body is worldBody
        body_index = next_next_step % self.rendered_step_count

        bound_checked_index = next_next_step % self.n_steps
        pos = np.copy(self.terrain_info[bound_checked_index, 0:3])
        #pos[2] -= 20
        phi, x_tilt, y_tilt = self.terrain_info[bound_checked_index, 3:6]
        quaternion = self._p.getQuaternionFromEuler([x_tilt, y_tilt, phi])
        self.steps[body_index].set_position(pos=pos, quat=quaternion)
        self.targets = self.delta_to_k_targets(k=self.lookahead)

        # make phantom
        if self.make_phantoms_yes and self.next_step_index % 2 == 0:
            phantom = self.phantoms[self.current_phantom_idx]
            # set the phantom pose to current pose
            current_pose = self.robot.to_radians(self.robot.joint_angles)
            current_pos = self.robot.body_xyz
            current_orientation = self.robot.robot_body.pose().orientation()
            phantom.reset(pos=current_pos, pose=current_pose, quat=current_orientation)

            self.current_phantom_idx += 1
            self.current_phantom_idx %= 20

    def sample_next_next_step_1(self):
        pairs = np.indices(dimensions=(self.yaw_sample_size, self.pitch_sample_size))
        self.yaw_pitch_prob /= self.yaw_pitch_prob.sum()
        inds = self.np_random.choice(np.arange(self.yaw_sample_size * self.pitch_sample_size),
                                     p=self.yaw_pitch_prob.reshape(-1), size=1, replace=False)

        inds = pairs.reshape(2, self.yaw_sample_size * self.pitch_sample_size)[:, inds].squeeze()
        # print(self.yaw_pitch_prob, inds)
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2

        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)

        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = self.np_random.uniform(*self.r_range)

        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def sample_next_next_step_2(self):
        pairs = np.indices(dimensions=(self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size))
        self.yaw_pitch_r_prob /= self.yaw_pitch_r_prob.sum()
        inds = self.np_random.choice(np.arange(self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size),
                                     p=self.yaw_pitch_r_prob.reshape(-1), size=1, replace=False)

        inds = pairs.reshape(3, self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size)[:, inds].squeeze()
        # print(self.yaw_pitch_prob, inds)
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2
        dr = self.r_samples[inds[2]]

        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)

        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = dr

        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)

    def sample_next_next_step(self):
        pairs = np.indices(dimensions=(
            self.yaw_sample_size, self.pitch_sample_size, self.r_sample_size, self.x_tilt_sample_size,
            self.y_tilt_sample_size))
        self.yaw_pitch_r_tilt_prob /= self.yaw_pitch_r_tilt_prob.sum()
        inds = self.np_random.choice(np.arange(
            self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size * self.x_tilt_sample_size * self.y_tilt_sample_size),
            p=self.yaw_pitch_r_tilt_prob.reshape(-1), size=1, replace=False)

        inds = pairs.reshape(5,
                             self.yaw_sample_size * self.pitch_sample_size * self.r_sample_size * self.x_tilt_sample_size * self.y_tilt_sample_size)[
               :, inds].squeeze()
        # print(self.yaw_pitch_prob, inds)
        yaw = self.yaw_samples[inds[0]]
        pitch = self.pitch_samples[inds[1]] + np.pi / 2
        dr = self.r_samples[inds[2]]
        x_tilt = self.x_tilt_samples[inds[3]]
        y_tilt = self.y_tilt_samples[inds[4]]

        self.next_pitch = self.next_next_pitch
        self.next_yaw = self.next_next_yaw
        self.next_dr = np.copy(self.next_next_dr)
        self.next_x_tilt = np.copy(self.next_next_x_tilt)
        self.next_y_tilt = np.copy(self.next_next_y_tilt)

        self.next_next_pitch = pitch
        self.next_next_yaw = yaw
        self.next_next_dr = dr
        self.next_next_x_tilt = x_tilt
        self.next_next_y_tilt = y_tilt

        self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr, x_tilt=x_tilt,
                                         y_tilt=y_tilt)

    def get_temp_state(self):
        obs = self.get_state()
        target = self.delta_to_k_targets(k=self.lookahead)
        return np.concatenate((obs, target.flatten()))

    def create_temp_states_1(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    actual_pitch = np.pi / 2 - pitch
                    # print(actual_pitch / np.pi * 180)
                    # self.set_next_step_location(actual_pitch, yaw, 0.7)
                    self.set_next_next_step_location(actual_pitch, yaw, 0.7)
                    temp_state = self.get_temp_state()
                    temp_states.append(temp_state)
            # self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def create_temp_states_2(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    for r in self.fake_r_samples:
                        actual_pitch = np.pi / 2 - pitch
                        # self.set_next_step_location(actual_pitch, yaw, r)
                        self.set_next_next_step_location(actual_pitch, yaw, r)
                        temp_state = self.get_temp_state()
                        temp_states.append(temp_state)
            # self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def create_temp_states(self):
        if self.update_terrain:
            temp_states = []
            for yaw in self.fake_yaw_samples:
                for pitch in self.fake_pitch_samples:
                    for r in self.fake_r_samples:
                        for x_tilt in self.fake_x_tilt_samples:
                            for y_tilt in self.fake_y_tilt_samples:
                                actual_pitch = np.pi / 2 - pitch
                                self.set_next_next_step_location(actual_pitch, yaw, r, x_tilt, y_tilt)
                                temp_state = self.get_temp_state()
                                temp_states.append(temp_state)
            # self.set_next_step_location(self.next_pitch, self.next_yaw, self.next_dr)
            self.set_next_next_step_location(self.next_next_pitch, self.next_next_yaw, self.next_next_dr,
                                             self.next_next_x_tilt, self.next_next_y_tilt)
            ret = np.stack(temp_states)
        else:
            ret = self.temp_states
        return ret

    def set_next_next_step_location(self, pitch, yaw, dr, x_tilt=0, y_tilt=0):
        next_step_xyz = self.terrain_info[self.next_step_index]
        bound_checked_index = (self.next_step_index + 1) % self.n_steps

        dr = dr
        base_phi = self.base_phi[bound_checked_index]
        base_yaw = self.terrain_info[self.next_step_index, 3]

        dx = dr * np.sin(pitch) * np.cos(yaw + base_phi)
        # clip to prevent overlapping
        dx = np.sign(dx) * min(max(abs(dx), self.step_radius * 2.5), self.r_range[1])
        dy = dr * np.sin(pitch) * np.sin(yaw + base_phi)

        matrix = np.array([
            [np.cos(base_yaw), -np.sin(base_yaw)],
            [np.sin(base_yaw), np.cos(base_yaw)]
        ])

        dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

        x = next_step_xyz[0] + dxy[0]
        y = next_step_xyz[1] + dxy[1]
        z = next_step_xyz[2] + dr * np.cos(pitch)

        self.terrain_info[bound_checked_index, 0] = x
        self.terrain_info[bound_checked_index, 1] = y
        self.terrain_info[bound_checked_index, 2] = z
        self.terrain_info[bound_checked_index, 3] = yaw + base_yaw
        self.terrain_info[bound_checked_index, 4] = x_tilt
        self.terrain_info[bound_checked_index, 5] = y_tilt

    def set_next_step_location(self, pitch, yaw, dr, x_tilt=0, y_tilt=0):
        next_step_xyz = self.terrain_info[self.next_step_index - 1]
        bound_checked_index = (self.next_step_index) % self.n_steps

        dr = dr
        base_phi = self.base_phi[bound_checked_index]
        base_yaw = self.terrain_info[self.next_step_index - 1, 3]

        dx = dr * np.sin(pitch) * np.cos(yaw + base_phi)
        # clip to prevent overlapping
        dx = np.sign(dx) * min(max(abs(dx), self.step_radius * 2.5), self.r_range[1])
        dy = dr * np.sin(pitch) * np.sin(yaw + base_phi)

        matrix = np.array([
            [np.cos(base_yaw), -np.sin(base_yaw)],
            [np.sin(base_yaw), np.cos(base_yaw)]
        ])

        dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

        x = next_step_xyz[0] + dxy[0]
        y = next_step_xyz[1] + dxy[1]
        z = next_step_xyz[2] + dr * np.cos(pitch)

        self.terrain_info[bound_checked_index, 0] = x
        self.terrain_info[bound_checked_index, 1] = y
        self.terrain_info[bound_checked_index, 2] = z
        self.terrain_info[bound_checked_index, 3] = yaw + base_yaw
        self.terrain_info[bound_checked_index, 4] = x_tilt
        self.terrain_info[bound_checked_index, 5] = y_tilt

    def update_sample_prob_1(self, sample_prob):
        if self.update_terrain:
            self.yaw_pitch_prob = sample_prob
            self.update_terrain_info()

    def update_sample_prob_2(self, sample_prob):
        if self.update_terrain:
            self.yaw_pitch_r_prob = sample_prob
            self.update_terrain_info()

    def update_sample_prob(self, sample_prob):
        if self.update_terrain:
            self.yaw_pitch_r_tilt_prob = sample_prob
            self.update_terrain_info()

    def update_curriculum_1(self, curriculum):
        self.yaw_pitch_prob *= 0
        self.yaw_pitch_prob[(self.yaw_sample_size - 1) // 2, (self.pitch_sample_size - 1) // 2] = 1
        # self.curriculum = min(curriculum, self.max_curriculum)
        # half_size = (self.sample_size-1)//2
        # if self.curriculum >= half_size:
        #    self.curriculum = half_size
        # self.yaw_pitch_prob *= 0
        # prob = 1.0 / (self.curriculum * 2 + 1)**2
        # window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        # self.yaw_pitch_prob[window, window] = prob

    def update_curriculum_2(self, curriculum):
        self.yaw_pitch_r_prob *= 0
        self.yaw_pitch_r_prob[(self.yaw_sample_size - 1) // 2, (self.pitch_sample_size - 1) // 2, 0] = 1
        # self.curriculum = min(curriculum, self.max_curriculum)
        # half_size = (self.sample_size-1)//2
        # if self.curriculum >= half_size:
        #     self.curriculum = half_size
        # self.yaw_pitch_r_prob *= 0
        # prob = 1.0 / (self.curriculum * 2 + 1)**3
        # window = slice(half_size-self.curriculum, half_size+self.curriculum+1)
        # self.yaw_pitch_r_prob[window, window, 0:self.curriculum * 2 + 1] = prob

    def update_curriculum(self, curriculum):
        # self.yaw_pitch_r_tilt_prob *= 0
        # self.yaw_pitch_r_tilt_prob[(self.yaw_sample_size-1)//2, (self.pitch_sample_size-1)//2, 0, (self.x_tilt_sample_size-1)//2, (self.y_tilt_sample_size-1)//2] = 1
        self.curriculum = min(curriculum, self.max_curriculum)
        half_size = (self.sample_size - 1) // 2
        if self.curriculum >= half_size:
            self.curriculum = half_size
        self.yaw_pitch_r_tilt_prob *= 0
        prob = 1.0 / (self.curriculum * 2 + 1) ** 5
        window = slice(half_size - self.curriculum, half_size + self.curriculum + 1)
        self.yaw_pitch_r_tilt_prob[window, window, 0:self.curriculum * 2 + 1, window, window] = prob

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
                6,  # yaw, wx, wz
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
