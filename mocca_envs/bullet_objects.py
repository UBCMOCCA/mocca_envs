import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np

DEG2RAD = np.pi / 180


class VSphere:
    def __init__(self, bc, radius=None, pos=None, rgba=None):
        self._p = bc

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else rgba

        shape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            specularColor=[0.4, 0.4, 0],
        )

        self.id = self._p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=shape, basePosition=pos
        )
        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba

    def set_position(self, pos=None):

        pos = self._pos if pos is None else pos

        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=self._quat)

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id, -1, rgbaColor=rgba)
            self._rgba = t_rgba


class BaseStep:
    def __init__(self, bc, filename, scale, pos=None, quat=None):
        self._p = bc

        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        self.id = self._p.loadURDF(
            filename,
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=False,
            globalScaling=scale,
        )

        self._pos_offset = np.array(self._p.getBasePositionAndOrientation(self.id)[0])

        for link_id in range(-1, self._p.getNumJoints(self.id)):
            self._p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=1.0,
                restitution=0.1,
                contactStiffness=30000,
                contactDamping=1000,
            )

        self.base_id = -1
        self.cover_id = 0

    def set_position(self, pos=None, quat=None):
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        self._p.resetBasePositionAndOrientation(
            self.id, posObj=pos + self._pos_offset, ornObj=quat
        )


class Pillar(BaseStep):
    def __init__(self, bc, radius, pos=None, quat=None):
        filename = os.path.join(current_dir, "data", "objects", "steps", "pillar.urdf")
        super().__init__(bc, filename, radius, pos, quat)


class Plank(BaseStep):
    def __init__(self, bc, width, pos=None, quat=None):
        filename = os.path.join(current_dir, "data", "objects", "steps", "plank.urdf")
        super().__init__(bc, filename, 2 * width, pos, quat)


class LargePlank(BaseStep):
    def __init__(self, bc, width, pos=None, quat=None):
        filename = os.path.join(
            current_dir, "data", "objects", "steps", "plank_large.urdf"
        )
        super().__init__(bc, filename, 2 * width, pos, quat)


class Rectangle:
    def __init__(
        self, bc, hdx, hdy, hdz, mass=0.0, lateral_friction=0.8, pos=None, rgba=None
    ):
        self._p = bc

        dims = np.array([hdx, hdy, hdz], dtype=np.float32)

        pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos
        rgba = (55 / 255, 55 / 255, 55 / 255, 1) if rgba is None else rgba

        self._pos = pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])

        box_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=dims)
        box_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=dims,
            rgbaColor=rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.id = self._p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_vshape,
            basePosition=self._pos,
        )

    def set_position(self, pos=None, quat=None):

        pos = self._pos if pos is None else pos
        quat = self._quat if quat is None else quat

        self._pos = pos
        self._quat = quat

        self._p.resetBasePositionAndOrientation(
            self.id, posObj=self._pos, ornObj=self._quat
        )


class Sofa:
    """ Just a chair with cushion on top """

    def __init__(self, bc, hdx, hdy, hdz, mass=0.0, lateral_friction=0.8, pos=None):
        self._p = bc

        pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos

        self._pos = pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])

        box_dims = np.array([hdx, hdy, 4 * hdz / 5], dtype=np.float32)
        cushion_dims = np.array([hdx, hdy, hdz / 5], dtype=np.float32)

        self._offset = np.array([0, 0, cushion_dims[2]], dtype=np.float32)

        box_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=box_dims)
        box_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=box_dims,
            rgbaColor=(88 / 255, 99 / 255, 110 / 255, 1),
            specularColor=(0.4, 0.4, 0),
        )

        cushion_shape = self._p.createCollisionShape(
            self._p.GEOM_BOX, halfExtents=cushion_dims
        )
        cushion_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=cushion_dims,
            rgbaColor=(55 / 255, 66 / 255, 77 / 255, 1),
            specularColor=(0.4, 0.4, 0),
        )

        cushion_offset = np.array([0, 0, cushion_dims[2] + box_dims[2]])

        self.id = self._p.createMultiBody(
            baseMass=4 * mass / 5,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_vshape,
            basePosition=self._pos - self._offset,
            linkMasses=[mass / 5],
            linkCollisionShapeIndices=[cushion_shape],
            linkVisualShapeIndices=[cushion_vshape],
            linkPositions=[cushion_offset],
            linkOrientations=[(0, 0, 0, 1)],
            linkInertialFramePositions=[(0, 0, 0)],
            linkInertialFrameOrientations=[(0, 0, 0, 1)],
            linkParentIndices=[0],
            linkJointTypes=[self._p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 1)],
        )

        # Add softness to cushion
        self._p.changeDynamics(
            self.id,
            0,
            lateralFriction=lateral_friction,
            restitution=0.5,
            contactStiffness=1000,
            contactDamping=1000,
        )


class Chair:
    def __init__(self, bc, angle=-30, pos=None, quat=None):
        self._p = bc

        filename = os.path.join(current_dir, "data", "objects", "chairs", "chair.urdf")
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        flags = (
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        self.id = self._p.loadURDF(
            filename,
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=False,
            flags=flags,
        )
        self.set_angle(angle)

        # Add some softness to prevent from blowing up
        for link_id in range(-1, self._p.getNumJoints(self.id)):
            self._p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=0.8,
                restitution=0.2,
                contactStiffness=30000,
                contactDamping=1000,
            )

    def set_angle(self, angle):
        angle *= DEG2RAD

        # Set angle so it doesn't apply too much force
        self._p.resetJointState(self.id, 0, targetValue=angle, targetVelocity=0)
        # Maintain angle
        self._p.setJointMotorControl2(
            self.id,
            0,
            controlMode=self._p.POSITION_CONTROL,
            targetPosition=angle,
            targetVelocity=0,
            positionGain=1.0,
            velocityGain=0.1,
            force=1000,
        )


class Bench:
    def __init__(self, bc, pos=None, quat=None):
        self._p = bc

        filename = os.path.join(current_dir, "data", "objects", "chairs", "bench.urdf")
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        flags = (
            self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        self.id = self._p.loadURDF(
            filename,
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=False,
            flags=flags,
        )

        # Add some softness to prevent from blowing up
        for link_id in range(-1, self._p.getNumJoints(self.id)):
            self._p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=0.8,
                restitution=0.2,
                contactStiffness=30000,
                contactDamping=1000,
            )
