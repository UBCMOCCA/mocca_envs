import numpy as np
import gym

from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Walker3D

from .motion_data import MotionData


class Walker3DMimicEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    joints_with_data = [
        "right_hip_y",
        "right_knee",
        "right_ankle",
        "left_hip_y",
        "left_knee",
        "left_ankle",
    ]

    def __init__(self, render=False):
        super(Walker3DMimicEnv, self).__init__(Walker3D, render)
        self.robot.set_base_pose(pose="running_start")
        self.motion_data = MotionData("biped3d_walk.txt")
        joint_names = [j.joint_name for j in self.robot.ordered_joints]
        self.joint_with_data_inds = [
            joint_names.index(jname) for jname in self.joints_with_data
        ]

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 1)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def get_obs(self):
        return np.concatenate([[self.phase], self.robot_state])

    def reset(self, ctime=None):
        self.done = False

        self._p.restoreState(self.state_id)
        self.ctime = self.np_random.uniform(0, self.motion_data.get_duration())

        self.phase, pos, quat, vel, ang_vel, jpos, jvel = self.motion_data.get_state(
            self.ctime
        )
        pos[0] = 0

        self.robot.base_joint_angles[self.joint_with_data_inds] = jpos
        self.robot.base_joint_speeds[self.joint_with_data_inds] = jvel

        self.robot.base_joint_angles

        self.robot_state = self.robot.reset(
            random_pose=False, pos=pos, quat=quat, vel=vel, ang_vel=ang_vel
        )

        # Reset camera
        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)

        return self.get_obs()

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()

        self.ctime += self.control_step
        self.phase, *pose = self.motion_data.get_state(self.ctime)
        reward = self.calc_base_reward(pose)

        self.robot_state = self.robot.calc_state(self.ground_ids)

        # Calculate done
        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        if height < 0.7:
            self.done = True

        if self.is_render:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)

        return self.get_obs(), reward, self.done, {}

    def calc_base_reward(self, kin_pose):
        _, _, kin_vel, _, kin_jpos, _ = kin_pose
        dyn_jpos = np.array(
            [self.robot.jdict[jname].get_position() for jname in self.joints_with_data],
            dtype=np.float32,
        )

        jpos_err = np.sum(np.square(np.subtract(dyn_jpos, kin_jpos)))
        bvel_err = np.sum(np.square(np.subtract(self.robot.body_vel, kin_vel)))

        jpos_reward = np.exp(-4 * jpos_err)
        bvel_reward = np.exp(-10 * bvel_err)

        return 0.6 * jpos_reward + 0.4 * bvel_reward

    # def get_mirror_indices(self):

    #     action_dim = self.robot.action_space.shape[0]
    #     # _ + 6 accounting for global
    #     right = self.robot._right_joint_indices + 6
    #     # _ + action_dim to get velocities, 48 is right foot contact
    #     right = np.concatenate((right, right + action_dim, [48]))
    #     # Do the same for left, except using 49 for left foot contact
    #     left = self.robot._left_joint_indices + 6
    #     left = np.concatenate((left, left + action_dim, [49]))

    #     # Used for creating mirrored observations
    #     # 2:  vy
    #     # 4:  roll
    #     # 6:  abdomen_z pos
    #     # 8:  abdomen_x pos
    #     # 27: abdomen_z vel
    #     # 29: abdomen_x vel
    #     # 50: sin(-a) = -sin(a)
    #     negation_obs_indices = np.array([2, 4, 6, 8, 27, 29, 50], dtype=np.int64)
    #     right_obs_indices = right
    #     left_obs_indices = left

    #     # Used for creating mirrored actions
    #     negation_action_indices = self.robot._negation_joint_indices
    #     right_action_indices = self.robot._right_joint_indices
    #     left_action_indices = self.robot._left_joint_indices

    #     return (
    #         negation_obs_indices,
    #         right_obs_indices,
    #         left_obs_indices,
    #         negation_action_indices,
    #         right_action_indices,
    #         left_action_indices,
    #     )
