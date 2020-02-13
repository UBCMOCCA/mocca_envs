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
        self.step_radius = 0.08
        self.rendered_step_count = 4

        super().__init__(Monkey3D, render)
        self.robot.set_base_pose(pose="monkey_start")
        self.robot.base_position = (0, 0, self.initial_height)

        # Robot settings
        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        # Env settings
        self.n_steps = 32
        self.lookahead = 2
        self.next_step_index = 2
        self.stop_frames = 2

        # Terrain info
        self.pitch_limit = 25
        self.yaw_limit = 0
        self.r_range = np.array([0.7, 0.8])
        self.terrain_info = np.zeros((self.n_steps, 4))

        # robot_state + (2 targets) * (x, y, z) + 1 (time)
        robot_obs_dim = self.robot.observation_space.shape[0]
        high = np.inf * np.ones(robot_obs_dim + self.lookahead * 3 + 1)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # torques + 2 for contact
        robot_act_dim = self.robot.action_space.shape[0]
        high = np.inf * np.ones(robot_act_dim + 2)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def generate_step_placements(self, n_steps=50, yaw_limit=30, pitch_limit=25):

        y_range = np.array([-yaw_limit, yaw_limit]) * DEG2RAD
        p_range = np.array([90 + pitch_limit, 90 + pitch_limit]) * DEG2RAD

        dr = self.np_random.uniform(*self.r_range, size=n_steps)
        dphi = self.np_random.uniform(*y_range, size=n_steps)
        dtheta = self.np_random.uniform(*p_range, size=n_steps)

        # special treatment for first step
        dphi[0] = 0
        dphi[1] = 0

        dphi = np.cumsum(dphi)

        x_ = dr * np.sin(dtheta) * np.cos(dphi)
        y_ = dr * np.sin(dtheta) * np.sin(dphi)
        z_ = dr * np.cos(dtheta)

        # make first step directly on hand that is in front

        i = np.argmin(self.robot.feet_xyz[:, 0])
        x_[0] = self.robot.feet_xyz[i, 0]
        y_[0] = self.robot.feet_xyz[i, 1]
        z_[0] = (
            self.robot.feet_xyz[i, 2] - self.initial_height + self.step_radius + 0.05
        )

        j = np.argmax(self.robot.feet_xyz[:, 0])
        x_[1] = self.robot.feet_xyz[j, 0] - x_[0]
        y_[1] = self.robot.feet_xyz[j, 1] - y_[0]
        z_[1] = self.robot.feet_xyz[j, 2] - self.robot.feet_xyz[i, 2]

        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + self.initial_height

        self.swing_leg = i
        for index, (x1, y1, z1, k) in enumerate(zip(x[:2], y[:2], z[:2], [i, j])):
            id = self._p.createConstraint(
                self.robot.id,
                self.robot.feet[k].bodyPartIndex,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=self._p.JOINT_POINT2POINT,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=(x1, y1, z1),
            )
            self._p.changeConstraint(id, maxForce=100)
            self.holding_constraint_id[index] = id

        return np.stack((x, y, z, dphi), axis=1)

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

        mid = self.terrain_info[:, 0:3]
        yaw = self.terrain_info[:, 3]
        dir = np.stack((np.sin(yaw), np.cos(yaw), yaw * 0), axis=1)
        self.terrain_start_end = np.stack(
            (mid - dir * self.bar_length / 2, mid + dir * self.bar_length / 2)
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

    def reset(self):
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.done = False
        self.target_reached_count = 0
        self.holding = np.ones(len(self.robot.feet_xyz))

        if hasattr(self, "holding_constraint_id"):
            for id in self.holding_constraint_id:
                if id != -1:
                    self._p.removeConstraint(int(id))

        self.holding_constraint_id = np.array([-1, -1])
        self.free_fall_count = 0

        # start at 2 because first 2 are already in contact
        self.next_step_index = 2

        # self._p.restoreState(self.state_id)

        self.robot_state = self.robot.reset(random_pose=False)
        self.randomize_terrain()
        self.calc_feet_state()

        # Reset camera
        if self.is_rendered:
            self.camera.lookat(self.robot.body_xyz)

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        # Order is important because walk_target is set up above
        self.calc_potential()

        time = self.stop_frames - self.target_reached_count
        state = np.concatenate((self.robot_state, self.targets.flatten(), [time]))

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        return state

    def step(self, action):
        base_action = action[:-2]
        self.holding = action[-2:] > 0

        self.robot.apply_action(base_action)
        self.scene.global_step()

        # Don't calculate the contacts for now
        self.robot_state = self.robot.calc_state(contact_object_ids=None)
        self.calc_env_state(base_action)

        reward = self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus - 0 * self.speed_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        time = self.stop_frames - self.target_reached_count
        state = np.concatenate((self.robot_state, self.targets.flatten(), [time]))

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
        if not -40 < self.robot.body_rpy[1] * RAD2DEG < 40:
            self.posture_penalty = abs(self.robot.body_rpy[1])

        if not -40 < self.robot.body_rpy[0] * RAD2DEG < 40:
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

        self.tall_bonus = 2.0
        self.done = self.done or (self.free_fall_count > 30)

    def calc_feet_state(self):
        # Calculate contact separately for step
        target_cover_index = self.next_step_index % self.rendered_step_count
        next_step = self.steps[target_cover_index]

        self.foot_dist_to_target = np.array([0.0, 0.0])

        p_xyz = self.terrain_info[self.next_step_index, [0, 1, 2]]
        self.target_reached = False
        for i, f in enumerate(self.robot.feet):
            self.robot.feet_xyz[i] = f.pose().xyz()

            delta = self.robot.feet_xyz[i] - p_xyz
            distance = (delta[0] ** 2 + delta[1] ** 2) ** (1 / 2)
            self.foot_dist_to_target[i] = distance

            points = self._p.getClosestPoints(
                self.robot.id,
                next_step.id,
                self.step_radius * 1.5,
                f.bodyPartIndex,
                next_step.cover_id,
            )
            in_contact = False
            if len(points) > 0:
                p = points[0][5]
                in_contact = i == self.swing_leg

            constraint_id = self.holding_constraint_id[i]
            if in_contact:
                self.target_reached = True

                if self.holding[i] and constraint_id == -1:
                    id = self._p.createConstraint(
                        self.robot.id,
                        f.bodyPartIndex,
                        childBodyUniqueId=-1,
                        childLinkIndex=-1,
                        jointType=self._p.JOINT_POINT2POINT,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=p,
                    )
                    self._p.changeConstraint(id, maxForce=100)
                    self.holding_constraint_id[i] = id

            if not self.holding[i] and constraint_id != -1:
                self._p.removeConstraint(int(constraint_id))
                self.holding_constraint_id[i] = -1

        # At least one foot is on the plank
        if self.target_reached:
            self.target_reached_count += 1

            # Make target stationary for a bit
            if self.target_reached_count >= self.stop_frames:
                self.next_step_index += 1
                self.target_reached_count = 0
                # hard-code alternating swing leg
                self.swing_leg = (self.swing_leg + 1) % 2
                self.update_steps()

            # Prevent out of bound
            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1

        # # Swing foot
        # if (self.holding_constraint_id == -1).all():
        #     p_xyz = self.terrain_info[self.next_step_index, 0:3]
        #     distance = np.linalg.norm(p_xyz - self.robot.feet_xyz, axis=-1)
        #     self.swing_leg = np.argmax(distance)
        # else:
        #     self.swing_leg = np.argmin(self.holding_constraint_id)

        # Foot contact
        holding = (self.holding_constraint_id > -1).astype(np.float32)
        self.robot.feet_contact[:] = holding

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

        # use contact to detect done
        mask = float(np.sum(self.holding_constraint_id) == -2)
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

        # _ + 6 accounting for global
        right = self.robot._right_joint_indices + global_dim
        # _ + robot_act_dim to get velocities, right foot contact
        right = np.concatenate((right, right + robot_act_dim, [robot_obs_dim - 2]))
        # Do the same for left
        left = self.robot._left_joint_indices + global_dim
        left = np.concatenate((left, left + robot_act_dim, [robot_obs_dim - 1]))

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
        right_action_indices = np.concatenate((self.robot._right_joint_indices, [23]))
        left_action_indices = np.concatenate((self.robot._left_joint_indices, [24]))

        return (
            negation_obs_indices,
            right_obs_indices,
            left_obs_indices,
            negation_action_indices,
            right_action_indices,
            left_action_indices,
        )
