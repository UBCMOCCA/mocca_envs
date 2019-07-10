import os
import json
import numpy as np

from pyquaternion import Quaternion


def _get_orientation_from_quaternion(elements):
    q = Quaternion(elements)
    return q.get_axis()[2] * q.angle


def _get_orientations_from_quaternion_seq(seq):
    return [_get_orientation_from_quaternion(v) for v in seq]


class MotionData(object):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "humanoid", "mocap")

    def __init__(self, motion_file):
        file_path = os.path.join(self.path, motion_file)
        with open(file_path, "r") as mfile:
            data = json.load(mfile)

        if "Loop" in data and data["Loop"] is not True:
            # logger.warning('This data was not defined to loop around')
            pass
        if "Frames" not in data:
            raise ValueError(
                "The file `%s` does not conform to the correct TRL MoCap format"
                % file_path
            )

        frames = np.array(data["Frames"])
        self.duration = frames[:-1, 0].sum()
        self.dt = frames[0, 0]
        self.len = len(frames) - 1

        self.torso_pos = frames[:, 1:4]

        self.torso_pos[:, 2] += data.get("ZCompensation", 1.28)
        # self.torso_pos[:, 1] += data.get("YCompensation", 0.28)
        # self.torso_pos[:, 1] += data.get("YCompensation", -0.76)
        self.torso_pos[:, 1] = 0
        # skipping the root and torso_pos orientation (always assuming upright position for now)

        self.torso_ori = np.concatenate([frames[:, 5:8], frames[:, 4:5]], axis=-1)

        joints = frames[:, 12:]

        joints_2d = np.vstack(
            [
                _get_orientations_from_quaternion_seq(joints[:, :4]),
                joints[:, 4],
                _get_orientations_from_quaternion_seq(joints[:, 5:9]),
                _get_orientations_from_quaternion_seq(joints[:, 9:13]),
                joints[:, 13],
                _get_orientations_from_quaternion_seq(joints[:, 14:18]),
            ]
        ).transpose()
        joints_2d[:, [0, 3]] *= -1
        self.jpos = joints_2d

    def get_duration(self):
        return self.duration

    def get_state(self, t):
        phase = (t % self.duration) / self.duration
        i = min(int(phase * self.len), self.len - 1)

        alpha = phase * self.len - i

        pos = (1 - alpha) * self.torso_pos[i] + alpha * self.torso_pos[i + 1]
        ori = self.torso_ori[i]
        vel = (self.torso_pos[i + 1] - self.torso_pos[i]) / self.dt
        angvel = np.array([0, 0, 0])  # TODO: fix later
        jpos = (1 - alpha) * self.jpos[i] + alpha * self.jpos[i + 1]
        jvel = (self.jpos[i + 1] - self.jpos[i]) / self.dt
        return (t, pos, ori, vel, angvel, jpos, jvel)


if __name__ == "__main__":
    data = MotionData("biped3d_walk.txt")
    import ipdb

    ipdb.set_trace()

