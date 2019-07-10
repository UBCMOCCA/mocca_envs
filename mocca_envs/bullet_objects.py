import numpy as np


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


class Sphere:
    def __init__(self, bc, radius=None, mass=0, pos=None, rgba=None):
        self._p = bc

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else rgba

        vshape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            specularColor=[0.4, 0.4, 0],
        )

        cshape = self._p.createCollisionShape(self._p.GEOM_SPHERE, radius=radius)

        self.id = self._p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=cshape,
            baseVisualShapeIndex=vshape,
            basePosition=pos,
        )
        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba

    def get_position(self):
        return self._p.getBasePositionAndOrientation(self.id)[0]

    def set_position(self, pos=None):

        pos = self._pos if pos is None else pos

        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=self._quat)
        self._pos = tuple(pos)


class Pillar:
    def __init__(self, bc, radius, length, pos=None, c_rgba=None, p_rgba=None):
        self._p = bc

        length = length - 0.01
        pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos
        c_rgba = (55 / 255, 55 / 255, 55 / 255, 1) if c_rgba is None else c_rgba
        p_rgba = (88 / 255, 99 / 255, 110 / 255, 1) if p_rgba is None else p_rgba

        self._pos = pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._offset = np.array([0.0, 0.0, length / 2])

        body_shape = self._p.createCollisionShape(
            self._p.GEOM_CYLINDER, radius=radius, height=length
        )
        cover_shape = self._p.createCollisionShape(
            self._p.GEOM_CYLINDER, radius=radius, height=0.01
        )

        body_vshape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=p_rgba,
            specularColor=(0.4, 0.4, 0),
        )

        cover_vshape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=radius,
            length=0.01,
            rgbaColor=c_rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.body_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_vshape,
            basePosition=self._pos - self._offset,
        )
        self._p.changeDynamics(self.body_id, -1, lateralFriction=1.0, restitution=0.1)

        self.cover_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cover_shape,
            baseVisualShapeIndex=cover_vshape,
            basePosition=self._pos,
        )
        self._p.changeDynamics(self.cover_id, -1, lateralFriction=1.0, restitution=0.1)

    def set_position(self, pos=None, quat=None):

        pos = self._pos if pos is None else pos
        quat = self._quat if quat is None else quat

        self._pos = pos
        self._quat = quat

        self._p.resetBasePositionAndOrientation(
            self.body_id, posObj=self._pos - self._offset, ornObj=self._quat
        )

        self._p.resetBasePositionAndOrientation(
            self.cover_id, posObj=self._pos, ornObj=self._quat
        )


class Plank:
    def __init__(self, bc, xyz, pos=None, c_rgba=None, p_rgba=None):
        self._p = bc

        b_xyz = np.array([xyz[0], xyz[1] / 2, (xyz[2] - 0.01) / 2])
        c_xyz = np.concatenate((b_xyz[0:2], [0.01]))

        pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos
        c_rgba = (55 / 255, 55 / 255, 55 / 255, 1) if c_rgba is None else c_rgba
        p_rgba = (88 / 255, 99 / 255, 110 / 255, 1) if p_rgba is None else p_rgba

        self._pos = pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._offset = np.array([0.0, 0.0, b_xyz[2]])

        body_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=b_xyz)
        cover_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=c_xyz)

        body_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=b_xyz,
            rgbaColor=p_rgba,
            specularColor=(0.4, 0.4, 0),
        )

        cover_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=c_xyz,
            rgbaColor=c_rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.body_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_vshape,
            basePosition=self._pos - self._offset,
        )
        self._p.changeDynamics(self.body_id, -1, lateralFriction=1.0, restitution=0.1)

        self.cover_id = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cover_shape,
            baseVisualShapeIndex=cover_vshape,
            basePosition=self._pos,
        )
        self._p.changeDynamics(self.cover_id, -1, lateralFriction=1.0, restitution=0.1)

    def set_position(self, pos, quat):
        self._pos = pos
        self._quat = quat

        self._p.resetBasePositionAndOrientation(
            self.body_id, posObj=self._pos - self._offset, ornObj=self._quat
        )

        self._p.resetBasePositionAndOrientation(
            self.cover_id, posObj=self._pos, ornObj=self._quat
        )


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
