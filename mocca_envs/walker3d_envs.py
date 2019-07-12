import gym
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.bullet_objects import VSphere, Pillar, Plank, Rectangle, Sofa
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

    def __init__(self, render=False):
        super(Walker3DCustomEnv, self).__init__(Walker3D, render)
        self.robot.set_base_pose(pose="running_start")

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
        self.dist = self.np_random.uniform(3, 5)
        self.angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        self.stop_frames = self.np_random.choice([30.0, 60.0])

    def reset(self):
        self.done = False
        self.add_angular_progress = True
        self.randomize_target()

        self.walk_target = np.array(
            [self.dist * np.cos(self.angle), self.dist * np.sin(self.angle), 1.0]
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

        if self.close_count >= self.stop_frames:
            self.close_count = 0
            self.add_angular_progress = True
            self.randomize_target()
            delta = self.dist * np.array([np.cos(self.angle), np.sin(self.angle), 0.0])
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

