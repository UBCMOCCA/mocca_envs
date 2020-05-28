import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import gym
import numpy as np

from mocca_envs.bullet_utils import BodyPart, Joint

DEG2RAD = np.pi / 180


class Cassie:
    model_path = os.path.join(
        current_dir, "data", "cassie", "urdf", "cassie_collide.urdf"
    )
    base_position = (0.0, 0.0, 1.085)
    base_orientation = (0.0, 0.0, 0.0, 1.0)

    base_joint_angles = [
        # left:
        0.035615837,
        -0.01348790,
        0.391940848,
        -0.95086160,
        -0.08376049,
        1.305643634,
        -1.61174064,
        # right:
        0.035615837,
        -0.01348790,
        0.391940848,
        -0.95086160,
        -0.08376049,
        1.305643634,
        -1.61174064,
    ]

    rod_joint_angles = [-0.8967891835, 0.063947468, -0.8967891835, -0.063947468]

    power_coef = {
        "hip_abduction_left": 112.5,
        "hip_rotation_left": 112.5,
        "hip_flexion_left": 195.2,
        "knee_joint_left": 195.2,
        "knee_to_shin_right": 200,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_right": 200,  # not sure how to set, using PD instead of a constraint
        "toe_joint_left": 45.0,
        "hip_abduction_right": 112.5,
        "hip_rotation_right": 112.5,
        "hip_flexion_right": 195.2,
        "knee_joint_right": 195.2,
        "knee_to_shin_left": 200,  # not sure how to set, using PD instead of a constraint
        "ankle_joint_left": 200,  # not sure how to set, using PD instead of a constraint
        "toe_joint_right": 45.0,
    }
    joint_damping = [1, 1, 1, 1, 0.1, 0, 1, 1, 1, 1, 1, 0.1, 0, 1]

    powered_joint_inds = [0, 1, 2, 3, 6, 7, 8, 9, 10, 13]
    spring_joint_inds = [4, 11]

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.power = power
        self.rod_joints = {}
        print(self.power)

        self.parts = None
        self.jdict = None
        self.object_id = None
        self.ordered_joints = None
        self.robot_body = None
        self.foot_names = ["right_toe", "left_toe"]

        action_dim = 10
        high = np.ones(action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        state_dim = (action_dim + 4) * 2 + 6
        high = np.inf * np.ones(state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = (
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            | self._p.URDF_USE_INERTIA_FROM_FILE
        )
        self.object_id = (
            self._p.loadURDF(
                self.model_path,
                basePosition=self.base_position,
                baseOrientation=self.base_orientation,
                useFixedBase=False,
                flags=flags,
            ),
        )

        self.parse_joints_and_links(self.object_id)
        self.powered_joints = np.array(self.ordered_joints)[
            self.powered_joint_inds
        ].tolist()
        self.spring_joints = np.array(self.ordered_joints)[
            self.spring_joint_inds
        ].tolist()

        # Set Initial pose
        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=self.base_position, ornObj=self.base_orientation
        )

        self.reset_joint_positions(
            self.base_joint_angles, [0 for _ in self.base_joint_angles]
        )

        self._p.createConstraint(
            self.object_id[self.parts["left_tarsus"].bodyIndex],
            self.parts["left_tarsus"].bodyPartIndex,
            self.object_id[self.parts["left_achilles_rod"].bodyIndex],
            self.parts["left_achilles_rod"].bodyPartIndex,
            jointType=self._p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.22735404, 0.05761813, 0.00711836],
            childFramePosition=[0.254001, 0, 0],
            parentFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
            childFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
        )
        self._p.createConstraint(
            self.object_id[self.parts["right_tarsus"].bodyIndex],
            self.parts["right_tarsus"].bodyPartIndex,
            self.object_id[self.parts["right_achilles_rod"].bodyIndex],
            self.parts["right_achilles_rod"].bodyPartIndex,
            jointType=self._p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.22735404, 0.05761813, -0.00711836],
            childFramePosition=[0.254001, 0, 0],
            parentFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
            childFrameOrientation=[0.000000, 0.000000, 0.000000, 1.000000],
        )
        for part_name in [
            "left_achilles_rod",
            "left_achilles_rod_y",
            "right_achilles_rod",
            "right_achilles_rod_y",
        ]:
            self._p.setCollisionFilterGroupMask(
                self.object_id[self.parts[part_name].bodyIndex],
                self.parts[part_name].bodyPartIndex,
                0,
                0,
            )

    def reset_joint_positions(self, positions, velocities):
        for j, q, v in zip(self.ordered_joints, positions, velocities):
            j.reset_current_position(q, v)

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

        # We will overwrite this if a "pelvis" is found
        self.robot_body = BodyPart(self._p, "root", bodies, 0, -1)

        for i in range(len(bodies)):
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i],
                    j,
                    self._p.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == "pelvis":
                    self.robot_body = self.parts[part_name]

                joint = Joint(self._p, joint_name, bodies, i, j, torque_limit=0)

                if "achilles" in joint_name:
                    joint.reset_position(self.rod_joint_angles[len(self.rod_joints)], 0)
                    self.rod_joints[joint_name] = joint

                if joint_name[:5] != "fixed":
                    joint.set_torque_limit(self.power * self.power_coef[joint_name])
                    self.jdict[joint_name] = joint
                    self._p.changeDynamics(
                        bodies[i],
                        j,
                        jointDamping=self.joint_damping[len(self.ordered_joints)],
                    )
                    self.ordered_joints.append(self.jdict[joint_name])

    def make_robot_utils(self):
        # Make utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        # weight is the range of motion, thigh can move 90 degrees, etc
        weight = np.array([j.upperLimit - j.lowerLimit for j in self.ordered_joints])
        # bias is the angle corresponding to -1
        bias = np.array([j.lowerLimit for j in self.ordered_joints])
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def reset(self):
        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_xyz = np.zeros((len(self.foot_names), 3))
        self.initial_z = None
        state = self.calc_state()
        return state

    def apply_action(self, a):
        assert np.isfinite(a).all()
        # for n, j in enumerate(self.ordered_joints):
        for n, j in enumerate(self.powered_joints + self.spring_joints):
            # j.set_position(self.base_joint_angles[n])
            j.set_motor_torque(float(np.clip(a[n], -j.torque_limit, j.torque_limit)))

        # self.ordered_joints[4].set_position(0)
        # self.ordered_joints[11].set_position(0)
        # angles = self.to_radians(self.joint_angles)
        # self.ordered_joints[5].set_position(-angles[3] + 0.227)  # -q_3 + 13 deg
        # self.ordered_joints[12].set_position(-angles[10] + 0.227)  # -q_10 + 13 deg

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.rad_joint_angles = self.to_radians(self.joint_angles)
        self.joint_speeds = j[:, 1]
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()
        self.body_xyz = body_pose.xyz()
        self.body_angular_speed = self.robot_body.angular_speed()

        z = self.body_xyz[2]
        if self.initial_z is None:
            self.initial_z = z

        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        self.body_velocity = np.dot(rot, self.robot_body.speed())

        vx, vy, vz = self.body_velocity
        more = np.array([z - self.initial_z, vx, vy, vz, roll, pitch], dtype=np.float32)

        for i, p in enumerate(self.feet):
            # Need this to calculate done, might as well calculate it
            self.feet_xyz[i] = p.pose().xyz()

        return np.concatenate((more, self.joint_angles, self.joint_speeds))


class Cassie2D(Cassie):
    model_path = os.path.join(
        current_dir, "data", "cassie", "urdf", "cassie_collide_2d.urdf"
    )



class WalkerBase:

    mirrored = False

    def apply_action(self, a):
        assert np.isfinite(a).all()
        x = np.clip(a, -1, 1)
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.torque_limits[n] * float(x[n]))

    def cal_j_and_jv(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )
        j[:, 0] = self.to_radians(j[:, 0])
        #j[:, 1] = self.to_radians(j[:, 1])
        return j[:, 0], j[:, 1]

    def calc_state(self, contact_object_ids=None):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.joint_speeds = 0.1 * j[:, 1]  # Normalize
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()

        # parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()])
        # self.body_xyz = parts_xyz.mean(axis=0)
        # # pelvis z is more informative than mean z
        # self.body_xyz[2] = body_pose.xyz()[2]

        # Faster if we don't use true CoM
        self.body_xyz = body_pose.xyz()

        if self.initial_z is None:
            self.initial_z = self.body_xyz[2]

        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        self.body_vel = np.dot(rot, self.robot_body.speed())
        vx, vy, vz = self.body_vel

        wx, wy, wz = self.robot_body.angular_speed() / 10

        more = np.array(
            [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch],
            dtype=np.float32,
        )
        # vx, vy, vz = self.robot_body.speed()
        # more = np.array(
        #     [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz],
        #     dtype=np.float32,
        # )

        if contact_object_ids is not None:
            for i, f in enumerate(self.feet):
                self.feet_xyz[i] = f.pose().xyz()
                contact_ids = set((x[2], x[4]) for x in f.contact_list())
                if contact_object_ids & contact_ids:
                    self.feet_contact[i] = 1.0
                else:
                    self.feet_contact[i] = 0.0

        state = np.concatenate(
            (more, self.joint_angles, self.joint_speeds, self.feet_contact)
        )

        return np.clip(state, -5, +5)

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def load_robot_model(self, model_path, flags, root_link_name=None):
        self.object_id = self._p.loadMJCF(model_path, flags=flags)

        self.parse_joints_and_links(self.object_id)
        if root_link_name is not None:
            self.robot_body = self.parts[root_link_name]
        else:
            self.robot_body = BodyPart(self._p, "root", self.object_id, 0, -1)

        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))
        self.calc_torque_limits()

    def calc_torque_limits(self):
        self.torque_limits = self.power * np.array(
            [self.power_coef[j.joint_name] for j in self.ordered_joints]
        )

    def make_robot_utils(self):
        # utility functions for converting from normalized to radians and vice versa
        # Inputs are thetas (normalized angles) directly from observation
        # weight is the range of motion, thigh can move 90 degrees, etc
        weight = np.array([j.upperLimit - j.lowerLimit for j in self.ordered_joints])
        # bias is the angle corresponding to -1
        bias = np.array([j.lowerLimit for j in self.ordered_joints])
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

        for i in range(len(bodies)):
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i],
                    j,
                    self._p.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1]
                part_name = jointInfo[12]

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    self.jdict[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    self.ordered_joints.append(self.jdict[joint_name])

    def reset(self):
        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)
        self.initial_z = None

        robot_state = self.calc_state()
        return robot_state


class Walker3D(WalkerBase):

    foot_names = ["right_foot", "left_foot"]
    power_amplifier = 1.0
    power_coef = {
        "abdomen_z": 60*power_amplifier,#120,
        "abdomen_y": 80*power_amplifier,#160,
        "abdomen_x": 60*power_amplifier,#120,
        "right_hip_x": 80*power_amplifier,#160,
        "right_hip_z": 60*power_amplifier,#120,
        "right_hip_y": 100*power_amplifier,#200,
        "right_knee": 90*power_amplifier,#180,
        "right_ankle": 60*power_amplifier,#120,
        "left_hip_x": 80*power_amplifier,#160,
        "left_hip_z": 60*power_amplifier,#120,
        "left_hip_y": 100*power_amplifier,#200,
        "left_knee": 90*power_amplifier,#180,
        "left_ankle": 60*power_amplifier,#120,
        "right_shoulder_x": 60*power_amplifier,#120,
        "right_shoulder_z": 60*power_amplifier,#120,
        "right_shoulder_y": 50*power_amplifier,#100,
        "right_elbow": 60*power_amplifier,#120,
        "left_shoulder_x": 60*power_amplifier,#120,
        "left_shoulder_z": 60*power_amplifier,#120,
        "left_shoulder_y": 50*power_amplifier,#100,
        "left_elbow": 60*power_amplifier,#120,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 21
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.P = np.array([60,80,60,80,60,100,90,60,80,60,100,90,60,60,60,50,60,60,60,50,60])
        self.D = self.P / 10.0
        # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

    # def apply_action(self, a):
    #     assert np.isfinite(a).all()
    #     x = np.clip(a, -1, 1)
    #     for n, j in enumerate(self.ordered_joints):
    #         #print(j.current_relative_position())
    #         torque = self.P[n] * (float(x[n])-j.current_relative_position()[0]) - self.D[n] * j.current_relative_position()[1]
    #         #print(torque)
    #         j.set_motor_torque(torque)

    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        model_path = os.path.join(current_dir, "data", "custom", "walker3d.xml")
        root_link_name = None

        # Need to call this first to parse body
        super(Walker3D, self).load_robot_model(model_path, flags, root_link_name)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_position = (0, 0, 1.32)
        self.base_orientation = (0, 0, 0, 1)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 17, 18, 19, 20], dtype=np.int64
        )
        # abdomen_[x,z]
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)

        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

        # waist_part = self.parts['pelvis']
        # waist_body_index = waist_part.bodyIndex
        # waist_part_index = waist_part.bodyPartIndex
        # info = self._p.getDynamicsInfo(waist_part.bodies[waist_body_index], waist_part_index)
        # print('before mass', info[0])
        # self._p.changeDynamics(waist_part.bodies[waist_body_index], waist_part_index, mass=25)
        # info = self._p.getDynamicsInfo(waist_part.bodies[waist_body_index], waist_part_index)
        # print('after mass', info[0])

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset

        if pose == "running_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[10] = np.pi / 10  # Left leg back
            self.base_joint_angles[[13, 17]] = np.pi / 3  # Shoulder x
            self.base_joint_angles[[14]] = -np.pi / 6  # Right shoulder back
            self.base_joint_angles[[18]] = np.pi / 6  # Left shoulder forward
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            #print("something")
        elif pose == "zhaoming's pose":
            self.base_joint_angles[[13, 17]] = np.pi / 3  # Shoulder x
            self.base_joint_angles[[14]] = -np.pi / 6  # Right shoulder back
            self.base_joint_angles[[18]] = np.pi / 6  # Left shoulder forward
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
        elif pose == "sit":
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -np.pi / 2  # knee
        elif pose == "squat":
            angle = -20 * DEG2RAD
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -np.pi / 2  # knee
            self.base_joint_angles[[7, 12]] = angle  # ankles
            self.base_orientation = self._p.getQuaternionFromEuler([0, -angle, 0])
        elif pose == "crawl":
            self.base_joint_angles[[13, 17]] = np.pi / 2  # shoulder x
            self.base_joint_angles[[14, 18]] = np.pi / 2  # shoulder z
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -120 * DEG2RAD  # knee
            self.base_joint_angles[[7, 12]] = -20 * DEG2RAD  # ankles
            self.base_orientation = self._p.getQuaternionFromEuler([0, 90 * DEG2RAD, 0])
        elif pose == "mocap":
            self.base_joint_angles[0:3] = 0
            self.base_joint_angles[3:6] = np.array([0, 0, 0.5203672])
            self.base_joint_angles[6] = -0.2491160000
            self.base_joint_angles[7] = 0
            self.base_joint_angles[8:11] = 0
            self.base_joint_angles[11] = -0.3915320000
            self.base_joint_angles[12] = 0.2062253
            self.base_joint_angles[13:16] = np.array([np.pi / 3, 0, -0.2549215])
            self.base_joint_angles[16] = 0.1705710000
            self.base_joint_angles[17:20] = np.array([np.pi / 3, 0, 0.2230414])
            self.base_joint_angles[20] = 0.5813480000

    def reset(self, random_pose=True, pos=None, quat=None, pose=None, vel=None):
        base_joint_angles = np.copy(self.base_joint_angles)
        if self.np_random.rand() < 0:
            self.mirrored = True
            base_joint_angles[self._rl] = base_joint_angles[self._lr]
            base_joint_angles[self._negation_joint_indices] *= -1
        else:
            self.mirrored = False

        if random_pose:
            # Mirror initial pose

            # Add small deviations
            ds = self.np_random.uniform(low=-0.1, high=0.1, size=self.action_dim)
            ps = self.to_normalized(base_joint_angles + ds)
            ps = self.to_radians(np.clip(ps, -0.95, 0.95))

            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(ps[i], 0)
        else:
            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(base_joint_angles[i], 0)

        if pose is not None:
            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(pose[i], 0)

        pos = self.base_position if pos is None else pos
        quat = self.base_orientation if quat is None else quat
        vel = np.zeros(3) if vel is None else vel

        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=pos, ornObj=quat
        )

        self._p.resetBaseVelocity(self.object_id[0], linearVelocity=vel)

        return super(Walker3D, self).reset()


class Walker2D(WalkerBase):

    foot_names = ["foot", "foot_left"]

    power_coef = {
        "torso_joint": 100,
        "thigh_joint": 100,
        "leg_joint": 100,
        "foot_joint": 50,
        "thigh_left_joint": 100,
        "leg_left_joint": 100,
        "foot_left_joint": 50,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 7
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = self._p.MJCF_COLORS_FROM_FILE
        model_path = os.path.join(current_dir, "data", "custom", "walker2d.xml")

        root_link_name = "torso"
        # root_link_name = None
        
        super(Walker2D, self).load_robot_model(model_path, flags, root_link_name)

    def reset(self, random_pose=True):
        return super().reset()


class Crab2D(WalkerBase):

    foot_names = ["foot", "foot_left"]

    power_coef = {
        "thigh_left_joint": 100,
        "leg_left_joint": 100,
        "foot_left_joint": 50,
        "thigh_joint": 100,
        "leg_joint": 100,
        "foot_joint": 50,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 6
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        model_path = os.path.join(current_dir, "data", "custom", "crab2d.xml")
        root_link_name = "pelvis"
        super(Crab2D, self).load_robot_model(model_path, flags, root_link_name)

class Mike(WalkerBase):

    foot_names = ["right_foot", "left_foot"]
    power_amplifier = 1.0
    power_coef = {
        "abdomen_z": 0*power_amplifier,#120,
        "abdomen_y": 0*power_amplifier,#160,
        "abdomen_x": 0*power_amplifier,#120,
        "right_hip_x": 80*power_amplifier,#160,
        "right_hip_z": 60*power_amplifier,#120,
        "right_hip_y": 100*power_amplifier,#200,
        "right_knee": 90*power_amplifier,#180,
        "right_ankle": 60*power_amplifier,#120,
        "left_hip_x": 80*power_amplifier,#160,
        "left_hip_z": 60*power_amplifier,#120,
        "left_hip_y": 100*power_amplifier,#200,
        "left_knee": 90*power_amplifier,#180,
        "left_ankle": 60*power_amplifier,#120,
        "right_shoulder_x": 30*power_amplifier,#120,
        "right_shoulder_z": 30*power_amplifier,#120,
        "right_shoulder_y": 25*power_amplifier,#100,
        "right_elbow": 30*power_amplifier,#120,
        "left_shoulder_x": 30*power_amplifier,#120,
        "left_shoulder_z": 30*power_amplifier,#120,
        "left_shoulder_y": 25*power_amplifier,#100,
        "left_elbow": 30*power_amplifier,#120,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 21
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.P = np.array([60,80,60,80,60,100,90,60,80,60,100,90,60,60,60,50,60,60,60,50,60])
        self.D = self.P / 10.0

    # def calc_state(self, contact_object_ids=None):
    #     state = super().calc_state(contact_object_ids)
    #     quat = self.robot_body.pose().orientation()
    #     mat = np.array(self._p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
    #     glasses_xyz = self.body_xyz + np.matmul(mat, [0.25, 0, 0])
    #     self._p.resetBasePositionAndOrientation(self.glasses_id, glasses_xyz, quat)
        
    #     hardhat_xyz = self.body_xyz + np.matmul(mat, [0, -0.4, 0.15])
    #     self._p.resetBasePositionAndOrientation(self.hardhat_id, hardhat_xyz, quat)
        
    #     return state


    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        model_path = os.path.join(current_dir, "data", "custom", "mike.xml")
        root_link_name = None

        # Need to call this first to parse body
        super(Mike, self).load_robot_model(model_path, flags, root_link_name)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_position = (0, 0, 1.32)
        self.base_orientation = (0, 0, 0, 1)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 17, 18, 19, 20], dtype=np.int64
        )
        # abdomen_[x,z]
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)

        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

        waist_part = self.parts['waist']
        waist_body_index = waist_part.bodyIndex
        waist_part_index = waist_part.bodyPartIndex
        # info = self._p.getDynamicsInfo(waist_part.bodies[waist_body_index], waist_part_index)
        # print('before mass', info[0])
        self._p.changeDynamics(waist_part.bodies[waist_body_index], waist_part_index, mass=8)
        # info = self._p.getDynamicsInfo(waist_part.bodies[waist_body_index], waist_part_index)
        # print('after mass', info[0])

        # glasses_file = os.path.join(
        #     current_dir, "data", "misc", "glasses.obj"
        # )
        # glasses_shape = self._p.createVisualShape(
        #     shapeType=self._p.GEOM_MESH,
        #     fileName=glasses_file,
        #     # rgbaColor=[186 / 255, 186 / 255, 186 / 255, 1],
        #     # specularColor=[0, 0, 0],
        #     meshScale=[0.02, 0.02, 0.02],
        #     visualFrameOrientation=[0, 0, 1, 1],
        # )
        # self.glasses_id = self._p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=glasses_shape,
        #     useMaximalCoordinates=True,
        # )

        # hardhat_file = os.path.join(
        #     current_dir, "data", "misc", "hardhat.obj"
        # )
        # hardhat_shape = self._p.createVisualShape(
        #     shapeType=self._p.GEOM_MESH,
        #     fileName=hardhat_file,
        #     # rgbaColor=[186 / 255, 186 / 255, 186 / 255, 1],
        #     # specularColor=[0, 0, 0],
        #     meshScale=[0.02, 0.02, 0.02],
        # )
        # self.hardhat_id = self._p.createMultiBody(
        #     baseMass=0,
        #     baseVisualShapeIndex=hardhat_shape,
        #     useMaximalCoordinates=True,
        # )

        # f = lambda x: self._p.loadTexture(os.path.join(current_dir, "data", "misc", x))
        # hardhat_texture = f("hardhat.jpg")
        # self._p.changeVisualShape(self.hardhat_id, -1, textureUniqueId=hardhat_texture)

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset

        if pose == "running_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[10] = np.pi / 10  # Left leg back
            self.base_joint_angles[[13, 17]] = np.pi / 3  # Shoulder x
            self.base_joint_angles[[14]] = -np.pi / 6  # Right shoulder back
            self.base_joint_angles[[18]] = np.pi / 6  # Left shoulder forward
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            #print("something")

    def reset(self, random_pose=True, pos=None, quat=None, pose=None, vel=None):
        base_joint_angles = np.copy(self.base_joint_angles)
        if self.np_random.rand() < 0.5:
            self.mirrored = True
            base_joint_angles[self._rl] = base_joint_angles[self._lr]
            base_joint_angles[self._negation_joint_indices] *= -1
        else:
            self.mirrored = False

        if random_pose:
            # Mirror initial pose

            # Add small deviations
            ds = self.np_random.uniform(low=-0.1, high=0.1, size=self.action_dim)
            ps = self.to_normalized(base_joint_angles + ds)
            ps = self.to_radians(np.clip(ps, -0.95, 0.95))

            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(ps[i], 0)
        else:
            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(base_joint_angles[i], 0)

        if pose is not None:
            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(pose[i], 0)

        pos = self.base_position if pos is None else pos
        quat = self.base_orientation if quat is None else quat
        vel = np.zeros(3) if vel is None else vel

        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=pos, ornObj=quat
        )

        self._p.resetBaseVelocity(self.object_id[0], linearVelocity=vel)

        return super(Mike, self).reset()
