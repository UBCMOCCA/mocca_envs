import os
import gym
import pickle
import numpy as np

from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Cassie, Cassie2D
from mocca_envs import current_dir

from .loadstep import CassieTrajectory

# from mocca_envs.cassie.CassieParams import CassieParams
# from mocca_envs.cassie.robot_data import RobotData


class CassieEnv(EnvBase):

    control_step = 0.03
    llc_frame_skip = 100
    sim_frame_skip = 1

    ## PD gains:
    kp = np.array(
        [
            100,
            100,
            88,
            96,
            # 500,
            # 450,
            50,
            #
            100,
            100,
            88,
            96,
            # 500,
            # 450,
            50,
        ]
    )
    # kp = kp / 2
    # kp[[4, 9]] /= 2
    kd = kp / 10

    initial_velocity = [0, 0, 0]

    def __init__(self, render=False, planar=False, power_coef=1.0):
        """
            :params render: enables GUI rendering
            :params planar: constrains the robot movement to a 2D plane (rather than the full 3D motion)
            :params power_coef: multiplying factor that determines the torque limit
        """
        self.planar = planar
        robot_class = Cassie2D if planar else Cassie
        super(CassieEnv, self).__init__(robot_class, render, power=power_coef)

        high = np.inf * np.ones(self.robot.observation_space.shape[0] + 2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.robot.action_space

    def calc_potential(self, body_xyz):
        target_dist = (
            (self.walk_target[1] - body_xyz[1]) ** 2
            + (self.walk_target[0] - body_xyz[0]) ** 2
        ) ** (1 / 2)

        return -target_dist / self.control_step

    def resetJoints(self):
        self.robot.reset_joint_positions(self.base_angles(), self.base_velocities())

    def reset(self, istep=0):
        self.done = False
        self.istep = istep
        self.walk_target = np.array([1000.0, 0.0, 0.0])

        self._p.restoreState(self.state_id)
        self.resetJoints()
        self.robot.robot_body.reset_velocity(self.initial_velocity)

        self.robot_state = self.robot.reset()

        if self.is_render:
            self.camera.lookat(self.robot.body_xyz)

        self.potential = self.calc_potential(self.robot.body_xyz)
        return self.get_obs(self.robot_state)

    def pd_control(self, target_angles, target_speeds):
        self.istep += 1
        curr_angles = self.robot.rad_joint_angles[self.robot.powered_joint_inds]
        curr_speeds = self.robot.joint_speeds[self.robot.powered_joint_inds]

        perror = target_angles - curr_angles
        verror = np.clip(target_speeds - curr_speeds, -5, 5)

        # print(", ".join(["%4.1f" % s for s in perror]), end="   |   ")
        # print(", ".join(["%4.1f" % s for s in verror]))

        return self.kp * perror + self.kd * verror

    def base_angles(self):
        return np.array(self.robot.base_joint_angles)

    def base_velocities(self):
        return np.array([0 for _ in self.robot.base_joint_angles])

    def compute_rewards(self, action, torques):
        old_potential = self.potential
        self.potential = self.calc_potential(self.robot.body_xyz)
        progress = self.potential - old_potential

        tall_bonus = (
            2.0
            if self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2]) > 0.6
            else -1.0
        )

        dead = tall_bonus < 0

        return dead, {"AliveRew": tall_bonus, "ProgressRew": progress}

    def get_obs(self, robot_state):
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
        return state

    def step(self, a):
        target_angles = self.base_angles()[self.robot.powered_joint_inds]
        ## `knee_to_shin` and `ankle_joint` joints (both sides) do not have a motor
        ## we don't know how to set the constraints for them so we're using PD with fixed target instead
        target_angles += a

        torques = []
        done = False

        for _ in range(self.llc_frame_skip):
            target_speeds = target_angles * 0
            torque = self.pd_control(target_angles, target_speeds)
            torques.append(torque)
            self.robot.apply_action(torque)
            self.scene.global_step()
            robot_state = self.robot.calc_state()
            if self.is_render:
                self.camera.track(
                    pos=self.robot.body_xyz, smooth_coef=np.array([0.1, 0.01, 0.01])
                )
                self._handle_keyboard()
                done = done or self.done

        if not np.isfinite(robot_state).all():
            print("~INF~", robot_state)
            done = True

        dead, rewards = self.compute_rewards(a, torques)
        done = done or dead

        return self.get_obs(robot_state), sum(rewards.values()), done, rewards


class CassieMocapRewEnv(CassieEnv):
    def compute_rewards(self, action, torques):
        dead, rewards = super(CassieMocapRewEnv, self).compute_rewards(action, torques)
        # TODO: use self.initial_velocity
        vel_error = (self.robot.body_velocity[0] - 0.8) ** 2

        dyn_angles = self.robot.to_radians(self.robot.joint_angles)
        kin_angles = self.base_angles()

        joint_penalty = np.sqrt(
            np.sum((kin_angles - dyn_angles)[self.robot.powered_joint_inds] ** 2)
        )
        orientation_penalty = np.sum(np.power(self.robot.body_rpy, 2))
        com_penalty = np.sum(
            np.subtract(self.robot.body_xyz[1:], self.robot.base_position[1:]) ** 2
        )

        rewards = {}
        # rewards["AliveRew"] = 0
        # rewards["ProgressRew"] /= 4
        rewards["EnergyUseRew"] = -0.00005 * np.mean(np.power(torques, 2))
        rewards["DeviationRew"] = -0.05 * np.mean(np.power(action, 2))

        # TODO: add orientation reward if not 2D

        rewards["SpeedRew"] = 0.1 * np.exp(-4 * vel_error)
        rewards["ImitationRew"] = 0.65 * np.exp(-4 * joint_penalty)
        if not self.planar:
            rewards["OrientationRew"] = 0.1 * np.exp(-4 * orientation_penalty)
        rewards["CoMRew"] = 0.15 * np.exp(-4 * com_penalty)

        return dead, rewards


class CassieMocapEnv(CassieMocapRewEnv):
    mocap_path = os.path.join(current_dir, "data/cassie/mocap/", "cassie_step_data.pkl")
    mocap_frame_skip = 60
    mocap_cycle_length = 28
    initial_velocity = [0.8, 0, 0]

    def __init__(self, rsi=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rsi = rsi
        with open(self.mocap_path, "rb") as datafile:
            self.step_data = pickle.load(datafile)

    def reset(self):
        self.phase = (
            self.np_random.randint(0, self.mocap_cycle_length) if self.rsi else 0
        )
        return super().reset()

    def base_angles(self):
        return self.step_data[self.phase * self.mocap_frame_skip]  # TODO: fix time

    def base_velocities(self):
        return (
            self.mocap_frame_skip
            / self.control_step
            * (
                self.step_data[self.phase * self.mocap_frame_skip + 1]
                - self.step_data[self.phase * self.mocap_frame_skip]
            )
        )

    def step(self, action):
        self.phase = (self.phase + 1) % self.mocap_cycle_length
        return super().step(action)


class CassieOSUEnv(CassieMocapRewEnv):
    initial_velocity = [0, 0, 0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        high = np.inf * np.ones(80)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.traj = CassieTrajectory()
        # self.traj = RobotData()
        # self.traj = CassieParams()

    def reset(self, istep=None):
        if istep is None:
            istep = self.np_random.randint(0, 10000)  # 3971 # 5152
        obs = super().reset(istep=istep)

        t = (self.istep) * self.control_step / self.llc_frame_skip
        rod_joint_names = [
            "fixed_right_achilles_rod_joint_z",
            "fixed_right_achilles_rod_joint_y",
            "fixed_left_achilles_rod_joint_z",
            "fixed_left_achilles_rod_joint_y",
        ]
        # print(self.traj.rod_joint_angles(t))
        for jname, angle in zip(rod_joint_names, self.traj.rod_joint_angles(t)):
            self.robot.rod_joints[jname].reset_current_position(angle, 0)

        return obs

    def base_angles(self):
        t = (self.istep) * self.control_step / self.llc_frame_skip
        # print("pos time:", t, "step:", self.istep)
        # print(t)  # , self.traj.joint_angles(t))
        return self.traj.joint_angles(t)

    def base_velocities(self):
        t = self.istep * self.control_step / self.llc_frame_skip
        # print("vel time:", t, "step:", self.istep)
        return self.traj.joint_speeds(t)

    def get_obs(self, _):
        t = (self.istep + 1) * self.control_step / self.llc_frame_skip
        kin_state = self.traj.state(t)

        joint_angles = np.array(
            [j.get_position() for j in self.robot.ordered_joints], dtype=np.float32
        )
        # joint_angles[0] +=
        joint_speeds = np.array(
            [j.get_velocity() for j in self.robot.ordered_joints], dtype=np.float32
        )
        quaternion = self._p.getQuaternionFromEuler(self.robot.body_rpy)

        dyn_state = np.concatenate(
            [
                ## pos
                self.robot.body_xyz[1:],
                [quaternion[-1]],
                quaternion[:-1],
                # [1, 0, 0, 0],
                joint_angles,
                ## vel
                self.robot.body_velocity,
                self.robot.robot_body.angular_speed(),
                # [0, 0, 0],
                joint_speeds,
            ]
        )

        kin_state[0] = 0  # fix desired y to be exactly 0

        return np.concatenate([dyn_state, kin_state])


class CassieMocapPhaseEnv(CassieMocapEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kp[[0, 5]] *= 2
        self.kd[[0, 5]] *= 2
        high = np.concatenate([self.observation_space.high, [1]])
        self.observation_space = gym.spaces.Box(-1 * high, high, dtype=np.float32)

    def get_obs(self, robot_state):
        return np.concatenate(
            [super().get_obs(robot_state), [self.phase / self.mocap_cycle_length]]
        )

