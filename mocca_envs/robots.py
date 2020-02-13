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


class WalkerBase:

    mirrored = False

    def apply_action(self, a):
        assert np.isfinite(a).all()
        normalized = np.clip(a, -1, 1)

        self._p.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.ordered_joint_ids,
            controlMode=self._p.TORQUE_CONTROL,
            forces=self.ordered_joint_gains * normalized,
        )

    def calc_state(self, contact_object_ids=None):
        # j = np.array(
        #     [j.current_relative_position() for j in self.ordered_joints],
        #     dtype=np.float32,
        # )

        # Use pybullet's array version, should be faster
        j = self._p.getJointStates(self.id, self.ordered_joint_ids)
        j = np.array(j)[:, 0:2].astype(np.float32)

        self.joint_angles = self.to_normalized(j[:, 0])
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

        for i, f in enumerate(self.feet):
            self.feet_xyz[i] = f.pose().xyz()
            if contact_object_ids is not None:
                contact_ids = set((x[2], x[4]) for x in f.contact_list())
                if contact_object_ids & contact_ids:
                    self.feet_contact[i] = 1.0
                else:
                    self.feet_contact[i] = 0.0

        height = self.body_xyz[2] - np.min(self.feet_xyz[:, 2])
        more = np.array([height, vx, vy, vz, roll, pitch], dtype=np.float32)

        state = np.concatenate(
            (more, self.joint_angles, self.joint_speeds, self.feet_contact)
        )

        return np.clip(state, -5, +5)

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def load_robot_model(self, model_path, flags, root_link_name=None):
        self.object_id = self._p.loadMJCF(model_path, flags=flags)
        self.id = self.object_id[0]

        self.parse_joints_and_links(self.object_id)
        if root_link_name is not None:
            self.robot_body = self.parts[root_link_name]
        else:
            self.robot_body = BodyPart(self._p, "root", self.object_id, 0, -1)

        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))

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
        self.ordered_joint_ids = []
        self.ordered_joint_gains = []
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
                    gain = self.power * self.power_coef[joint_name]
                    self.jdict[joint_name] = Joint(
                        self._p, joint_name, bodies, i, j, gain
                    )
                    self.ordered_joints.append(self.jdict[joint_name])
                    self.ordered_joint_ids.append(j)
                    self.ordered_joint_gains.append(gain)

        # need to use it to calculate torques later
        self.ordered_joint_gains = np.array(self.ordered_joint_gains)
        self._zeros = [0 for _ in self.ordered_joint_ids]
        self._gains = [0.1 for _ in self.ordered_joint_ids]

    def reset(self):
        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)
        self.initial_z = None

        robot_state = self.calc_state()
        return robot_state

    def reset_joint_states(self, positions, velocities):

        for j, pos, vel in zip(self.ordered_joints, positions, velocities):
            self._p.resetJointState(
                j.bodies[j.bodyIndex], j.jointIndex, targetValue=pos, targetVelocity=vel
            )

        self._p.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.ordered_joint_ids,
            controlMode=self._p.POSITION_CONTROL,
            targetPositions=self._zeros,
            targetVelocities=self._zeros,
            positionGains=self._gains,
            velocityGains=self._gains,
            forces=self._zeros,
        )


class Walker3D(WalkerBase):

    foot_names = ["right_foot", "left_foot"]

    power_coef = {
        "abdomen_z": 60,
        "abdomen_y": 80,
        "abdomen_x": 60,
        "right_hip_x": 80,
        "right_hip_z": 60,
        "right_hip_y": 100,
        "right_knee": 90,
        "right_ankle": 60,
        "left_hip_x": 80,
        "left_hip_z": 60,
        "left_hip_y": 100,
        "left_knee": 90,
        "left_ankle": 60,
        "right_shoulder_x": 60,
        "right_shoulder_z": 60,
        "right_shoulder_y": 50,
        "right_elbow": 60,
        "left_shoulder_x": 60,
        "left_shoulder_z": 60,
        "left_shoulder_y": 50,
        "left_elbow": 60,
    }

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.power = power

        self.action_dim = 21
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if flags is None:
            flags = (
                self._p.MJCF_COLORS_FROM_FILE
                | self._p.URDF_USE_SELF_COLLISION
                | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            )

        if model_path is None:
            model_path = os.path.join(current_dir, "data", "custom", "walker3d.xml")

        # Need to call this first to parse body
        super(Walker3D, self).load_robot_model(model_path, flags, root_link_name)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_joint_speeds = np.zeros(self.action_dim)
        self.base_position = np.array([0, 0, 1.32])
        self.base_orientation = np.array([0, 0, 0, 1])
        self.base_velocity = np.array([0, 0, 0])
        self.base_angular_velocity = np.array([0, 0, 0])

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 17, 18, 19, 20], dtype=np.int64
        )
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)  # abdomen_[x,z]
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "running_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[10] = np.pi / 10  # Left leg back
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
            self.base_orientation = np.array(
                self._p.getQuaternionFromEuler([0, -angle, 0])
            )
        elif pose == "crawl":
            self.base_joint_angles[[13, 17]] = np.pi / 2  # shoulder x
            self.base_joint_angles[[14, 18]] = np.pi / 2  # shoulder z
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow
            self.base_joint_angles[[5, 10]] = -np.pi / 2  # hip
            self.base_joint_angles[[6, 11]] = -120 * DEG2RAD  # knee
            self.base_joint_angles[[7, 12]] = -20 * DEG2RAD  # ankles
            self.base_orientation = np.array(
                self._p.getQuaternionFromEuler([0, 90 * DEG2RAD, 0])
            )

    def reset(self, random_pose=True, pos=None, quat=None, vel=None, ang_vel=None):

        # Mirror initial pose
        if self.np_random.rand() < 0.5:
            self.base_joint_angles[self._rl] = self.base_joint_angles[self._lr]
            self.base_joint_angles[self._negation_joint_indices] *= -1
            self.base_orientation[0:3] *= -1

        if random_pose:
            # Add small deviations
            ds = self.np_random.uniform(low=-0.1, high=0.1, size=self.action_dim)
            ps = self.to_normalized(self.base_joint_angles + ds)
            ps = self.to_radians(np.clip(ps, -0.95, 0.95))
        else:
            ps = self.base_joint_angles

        self.reset_joint_states(ps, self.base_joint_speeds)

        pos = self.base_position if pos is None else pos
        quat = self.base_orientation if quat is None else quat

        self._p.resetBasePositionAndOrientation(
            self.object_id[0], posObj=pos, ornObj=quat
        )

        vel = self.base_velocity if vel is None else vel
        ang_vel = self.base_angular_velocity if ang_vel is None else ang_vel

        self.robot_body.reset_velocity(vel, ang_vel)

        return super(Walker3D, self).reset()


class Child3D(Walker3D):
    def __init__(self, bc):
        super().__init__(bc)
        self.power = 0.4

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if model_path is None:
            model_path = os.path.join(current_dir, "data", "custom", "child3d.xml")

        super().load_robot_model(model_path)
        self.base_position = (0, 0, 0.38)


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

    def __init__(self, bc, power=1.0):
        self._p = bc
        self.power = power

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
        root_link_name = "pelvis"
        super(Walker2D, self).load_robot_model(model_path, flags, root_link_name)


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


class Cassie2D(Cassie):
    model_path = os.path.join(
        current_dir, "data", "cassie", "urdf", "cassie_collide_2d.urdf"
    )


class Monkey3D(Walker3D):
    foot_names = ["right_hand", "left_hand"]

    power_coef = {
        "abdomen_z": 60,
        "abdomen_y": 60,
        "abdomen_x": 60,
        "right_hip_x": 50,
        "right_hip_z": 50,
        "right_hip_y": 50,
        "right_knee": 30,
        "right_ankle": 10,
        "left_hip_x": 50,
        "left_hip_z": 50,
        "left_hip_y": 50,
        "left_knee": 30,
        "left_ankle": 10,
        "right_shoulder_x": 120,
        "right_shoulder_z": 60,
        "right_shoulder_y": 120,
        "right_elbow_z": 60,
        "right_elbow_y": 120,
        "left_shoulder_x": 120,
        "left_shoulder_z": 60,
        "left_shoulder_y": 120,
        "left_elbow_z": 60,
        "left_elbow_y": 120,
    }

    def __init__(self, bc):
        super().__init__(bc)
        self.power = 0.7

        self.action_dim = 23
        high = np.ones(self.action_dim)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2
        high = np.inf * np.ones(self.state_dim)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_robot_model(self, model_path=None, flags=None, root_link_name=None):
        if model_path is None:
            model_path = os.path.join(current_dir, "data", "custom", "monkey3d.xml")

        super().load_robot_model(model_path)
        self.base_position = (0, 0, 0.7)

        # Need this to set pose and mirroring
        # hip_[x,z,y], knee, ankle, shoulder_[x,z,y], elbow
        self._right_joint_indices = np.array(
            [3, 4, 5, 6, 7, 13, 14, 15, 16, 17], dtype=np.int64
        )
        self._left_joint_indices = np.array(
            [8, 9, 10, 11, 12, 18, 19, 20, 21, 22], dtype=np.int64
        )
        self._negation_joint_indices = np.array([0, 2], dtype=np.int64)  # abdomen_[x,z]
        self._rl = np.concatenate((self._right_joint_indices, self._left_joint_indices))
        self._lr = np.concatenate((self._left_joint_indices, self._right_joint_indices))

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0  # reset
        self.base_orientation = np.array([0, 0, 0, 1])

        if pose == "monkey_start":
            self.base_joint_angles[[13]] = 20 * DEG2RAD  # shoulder x
            self.base_joint_angles[[14]] = -30 * DEG2RAD  # shoulder z
            self.base_joint_angles[[15]] = -130 * DEG2RAD  # shoulder y
            self.base_joint_angles[[18]] = 20 * DEG2RAD  # shoulder x
            self.base_joint_angles[[19]] = -30 * DEG2RAD  # shoulder z
            self.base_joint_angles[[20]] = -130 * DEG2RAD  # shoulder y
            self.base_joint_angles[[6, 11]] = -90 * DEG2RAD  # ankles
            self.base_joint_angles[[7, 12]] = -90 * DEG2RAD  # knees
            self.base_orientation = np.array(
                self._p.getQuaternionFromEuler([0, 0, -60 * DEG2RAD])
            )

    def reset(self, random_pose=True, pos=None, quat=None, vel=None, ang_vel=None):

        base_joint_angles = self.base_joint_angles.copy()
        base_joint_speeds = self.base_joint_speeds.copy()
        base_orientation = self.base_orientation.copy()

        if self.np_random.rand() < 0.5:
            self.mirrored = True
            base_joint_angles[self._rl] = base_joint_angles[self._lr]
            base_joint_angles[self._negation_joint_indices] *= -1
            base_joint_speeds[self._rl] = base_joint_speeds[self._lr]
            base_joint_speeds[self._negation_joint_indices] *= -1
            base_orientation[0:3] *= -1
        else:
            self.mirrored = False

        self.reset_joint_states(base_joint_angles, base_joint_speeds)

        pos = self.base_position if pos is None else pos
        quat = base_orientation if quat is None else quat
        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=quat)

        # call the WalkerBase reset, not Walker3D
        return super(Walker3D, self).reset()
