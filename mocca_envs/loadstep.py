import numpy as np
import gym
import math
import pybullet
import random
import os


class CassieTrajectory:
    filepath = os.path.join(
        os.path.dirname(__file__), "data", "cassie", "mocap", "stepdata.bin"
    )

    # pos_index = np.array(
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]
    # )
    pos_index = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 20, 21, 22, 23, 28, 29, 30, 34]
    )
    """
        1, 2                   -> pelvis pos y, z
        3, 4, 5, 6             -> pelvis orientation (quaternion)
        7                      -> hip abduction
        8                      -> hip yaw
        9                      -> hip rotation?
        10                     -> left_knee_spring
        11                     -> left_knee  ---> should be close to 0
        12                     -> left_ankle ---> should be -left_knee_spring + 13deg
        13                     -> left_toe

        21                     -> hip abduction
        22                     -> hip yaw
        23                     -> hip rotation?
        24                     -> left_knee_spring
        25                     -> left_knee
        26                     -> left_ankle
        27                     -> left_toe
    """
    rod_joints_index = np.array(
        [
            10,
            11,
            # 12, 13,
            24,
            25,
            # 26, 27
        ]
    )
    # vel_index = np.array(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25]
    # )
    vel_index = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 18, 19, 20, 21, 25, 26, 27, 31]
    )
    """
        0, 1, 2                -> translation
        3, 4, 5                -> orientation
        6                      -> hip abduction
        7                      -> hip yaw
        8                      -> hip rotation?
        9                      -> left_knee_spring
        10                     -> left_knee
        11                     -> left_ankle
        12                     -> left_toe

        19                     -> hip abduction
        20                     -> hip yaw
        21                     -> hip rotation?
        22                     -> left_knee_spring
        23                     -> left_knee
        24                     -> left_ankle
        25                     -> left_toe
    """
    # tarsus_v_index

    def __init__(self, data=None, filepath=None):
        if filepath is None:
            filepath = self.filepath
        if data is None:
            n = 1 + 35 + 32 + 10 + 10 + 10
            self.data = np.fromfile(self.filepath, dtype=np.double).reshape((-1, n))
        else:
            self.data = data
        self.time = self.data[:, 0]
        self.qpos = self.data[:, 1:36]
        self.qvel = self.data[:, 36:68]
        self.torque = self.data[:, 68:78]
        self.mpos = self.data[:, 78:88]
        self.mvel = self.data[:, 88:98]

    def __len__(self):
        return len(self.time)

    def max_time(self):
        return self.time[-1]

    def _sec_to_ind(self, t):
        tmax = self.time[-1]
        return int((t % tmax) / tmax * len(self.time))

    def state(self, t):
        """
            :params t: time in seconds
            :returns full state of size 40
        """
        i = self._sec_to_ind(t)
        return np.concatenate(
            (self.qpos[i][self.pos_index], self.qvel[i][self.vel_index])
        )

    def joint_angles(self, t):
        # signs = np.ones(14)
        # signs[]
        # 2 (y,z) + 4 (ori quaternion)
        return self.qpos[self._sec_to_ind(t)][self.pos_index[6:]]

    def rod_joint_angles(self, t):
        return self.qpos[self._sec_to_ind(t)][self.rod_joints_index]

    def joint_speeds(self, t):
        # 3 (x,y,z) + 3 (angular vel)
        return self.qvel[self._sec_to_ind(t)][self.vel_index[6:]]

    def action(self, t):
        tmax = self.time[-1]
        i = int((t % tmax) / tmax * len(self.time))
        return (self.mpos[i], self.mvel[i], self.torque[i])

    def sample(self):
        i = random.randrange(len(self.time))
        return (self.time[i], self.qpos[i], self.qvel[i])


def fix_rod_angles():
    # TODO: clean up, this is a mess ...

    print("IMPORTANT: set gravity to 0 and fix the base before using this")
    # TODO: do the above programatically

    env_name = "CassieOSUEnv-v0"
    env = gym.make(env_name, render=True).unwrapped
    traj = CassieTrajectory()

    for i in range(len(traj)):
        jpos = traj.qpos[i][traj.pos_index[6:]]
        env.robot.reset_joint_positions(jpos, jpos * 0)

        ###################
        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["right_tarsus"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        gdelta = np.matmul(mat, [-0.22735404, 0.05761813, -0.00711836])

        gpoint = np.add(
            env.unwrapped.robot.parts["right_tarsus"].current_position(), gdelta
        )

        jointglobal = env.unwrapped.robot.parts[
            "right_achilles_rod_y"
        ].current_position()

        lpoint = np.subtract(gpoint, jointglobal)

        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["right_thigh"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        lcpoint = np.matmul(mat.transpose(), lpoint)

        rot = math.atan2(lcpoint[1], lcpoint[0])

        env.unwrapped.robot.rod_joints[
            "fixed_right_achilles_rod_joint_z"
        ].reset_position(rot, 0)

        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["right_achilles_rod_y"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        lcpoint = np.matmul(mat.transpose(), lpoint)

        rot = math.atan2(lcpoint[2], lcpoint[0])
        env.unwrapped.robot.rod_joints[
            "fixed_right_achilles_rod_joint_y"
        ].reset_position(-rot, 0)
        ######################
        ###################
        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["left_tarsus"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        gdelta = np.matmul(mat, [-0.22735404, 0.05761813, 0.00711836])

        gpoint = np.add(
            env.unwrapped.robot.parts["left_tarsus"].current_position(), gdelta
        )

        jointglobal = env.unwrapped.robot.parts[
            "left_achilles_rod_y"
        ].current_position()

        lpoint = np.subtract(gpoint, jointglobal)

        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["left_thigh"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        lcpoint = np.matmul(mat.transpose(), lpoint)

        rot = math.atan2(lcpoint[1], lcpoint[0])

        env.unwrapped.robot.rod_joints[
            "fixed_left_achilles_rod_joint_z"
        ].reset_position(rot, 0)

        mat = env.unwrapped.robot._p.getMatrixFromQuaternion(
            env.unwrapped.robot.parts["left_achilles_rod_y"].current_orientation()
        )
        mat = np.array(mat).reshape((3, 3))
        lcpoint = np.matmul(mat.transpose(), lpoint)

        rot = math.atan2(lcpoint[2], lcpoint[0])
        env.unwrapped.robot.rod_joints[
            "fixed_left_achilles_rod_joint_y"
        ].reset_position(-rot, 0)
        ######################

        rod_joint_names = [
            "fixed_right_achilles_rod_joint_z",
            "fixed_right_achilles_rod_joint_y",
            "fixed_left_achilles_rod_joint_z",
            "fixed_left_achilles_rod_joint_y",
        ]

        for _ in range(20):
            env.scene.global_step()
            for jname in rod_joint_names:
                env.unwrapped.robot.rod_joints[jname].reset_position(
                    env.unwrapped.robot.rod_joints[jname].get_position(), 0
                )
            for j in env.unwrapped.robot.ordered_joints:
                j.reset_position(j.get_position(), 0)

        # no reason to change the angle of the feet
        mask = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])

        traj.data[i, 1 + traj.pos_index[6:][mask]] = [
            j.get_position() for j in env.unwrapped.robot.ordered_joints
        ][mask]
        traj.data[i, 1 + traj.rod_joints_index] = [
            env.unwrapped.robot.rod_joints[jname].get_position()
            for jname in rod_joint_names
        ]

    traj.data.tofile("loadstep2.bin")


if __name__ == "__main__":
    fix_rod_angles()
