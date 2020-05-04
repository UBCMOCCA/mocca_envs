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


class HeightField:
    def __init__(self, bc, height_field_size):
        self._p = bc
        self.height_field_size = height_field_size
        self.id = -1
        self.shape_id = -1

        texture_file = os.path.join(current_dir, "data", "misc", "checker_blue.png")
        self.texture_id = self._p.loadTexture(texture_file)
        self.texture_scaling = 20

        self.digitize_bins = 32
        self.total_length = 25.6

        # self.decorate_with_random_rocks()

    def decorate_with_random_rocks(self):
        if not hasattr(self, "data"):
            return

        hfield = self.data.reshape(self.height_field_size) * self.height_scale
        hrows = int(self.height_field_size[0] / 2)
        hcolumns = int(self.height_field_size[1] / 2)

        deltas = np.random.uniform(-2, 2, size=(10, 2))
        xys = (
            np.array(
                [
                    [-5, 4],
                    [-3, 4],
                    [0, 4],
                    [3, 4],
                    [4, 6],
                    [12, 10],
                    [12, 12],
                    [14, 14],
                    [14, 14],
                    [13, 6],
                    [2, 10],
                    [0, 10],
                    [6.5, 13],
                    [10, 13.5],
                    [5, 10],
                    [-3, 8],
                    [-4, 9],
                    [-5, 18],
                    [-3, 20],
                    [-1, 20],
                    [1, 20],
                    [3, 20],
                    [5, 20],
                    [7, 20],
                    [9, 18],
                ]
            )[:, None, :]
            + deltas[None, :, :]
        )
        xys = xys.reshape(np.prod(xys.shape[0:2]), 2)

        # VSphere(self._p, 1, [9, 20, 2])

        zs = hfield[
            (xys[:, 1] / (self.total_length / hrows) + hrows).astype(np.int32),
            (xys[:, 0] / (self.total_length / hcolumns) + hcolumns).astype(np.int32),
        ]
        xyzs = np.concatenate((xys, zs[:, None]), axis=-1)

        N = len(xyzs)

        rock_scales = np.array(
            [
                [0.4, 0.4, 0.4],
                [0.2, 0.2, 0.2],
                [0.02, 0.02, 0.02],
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
            ]
        )

        rock_files = [
            os.path.join(current_dir, "data", "misc", "stone_{}.obj".format(i))
            for i in range(1, 6)
        ]

        f = lambda x: self._p.loadTexture(os.path.join(current_dir, "data", "misc", x))
        rock_textures = [
            f("stone_1.png"),
            f("stone_2.jpg"),
            f("stone_3.jpg"),
            f("stone_5.png"),
        ]

        # Pybullet can't create too many visual shapes
        # create some shapes and reuse
        rock_shapes = [
            self._p.createVisualShape(
                shapeType=self._p.GEOM_MESH,
                fileName=rock_files[ri],
                rgbaColor=[186 / 255, 186 / 255, 186 / 255, 1],
                specularColor=[0, 0, 0],
                meshScale=rock_scales[ri] * np.random.uniform(1.2, 1.4),
                visualFrameOrientation=np.random.uniform(0, 1, 4),
            )
            for ri in np.random.randint(0, 5, 16)
        ]

        shape_indices = np.random.randint(0, len(rock_shapes), N)

        for i, si in enumerate(shape_indices):
            id = self._p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=rock_shapes[si],
                basePosition=xyzs[i],
                useMaximalCoordinates=True,
            )
            ti = rock_textures[np.random.randint(0, len(rock_textures))]
            self._p.changeVisualShape(id, -1, textureUniqueId=ti)

    def set_position(self, pos=None, quat=None):
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=quat)

    def reload(self, data=None, pos=(0, 0, 20), rendered=False):
        rows = self.height_field_size[0]
        cols = self.height_field_size[1]

        self.data = self.get_random_height_field() if data is None else data
        midpoint = int(self.data.shape[0] / 2 + rows / 2)
        height = self.data.max() / 2 - self.data[midpoint : midpoint + 1].mean()
        self.data -= self.data[midpoint : midpoint + 1].mean()

        if self.id >= 0:
            # Replace existing height field
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[1, 1, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=self.data,
                numHeightfieldRows=rows,
                numHeightfieldColumns=cols,
                replaceHeightfieldIndex=self.shape_id,
            )

        else:
            # Create if it's the first time
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[1, 1, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=self.data,
                numHeightfieldRows=rows,
                numHeightfieldColumns=cols,
            )

            self.id = self._p.createMultiBody(0, self.shape_id, -1, (0, 0, height))

            self._p.changeDynamics(self.id, -1, lateralFriction=0.7, restitution=0.2)

        if rendered:
            self._p.changeVisualShape(
                self.id,
                -1,
                textureUniqueId=self.texture_id,
                rgbaColor=[1, 1, 1, 1],
                specularColor=[0, 0, 0],
            )

    def get_random_height_field(self, rng=None):
        num_peaks = 256

        rng = np.random if rng is None else rng

        # peak scale
        scale = rng.normal(3, 1, size=num_peaks)[:, None, None]

        # peak positions
        x0 = rng.uniform(-1, 1, size=num_peaks)[:, None, None]
        y0 = rng.uniform(-1, 1, size=num_peaks)[:, None, None]

        # peak spread
        xs = rng.uniform(0.01, 0.02, size=num_peaks)[:, None, None]
        ys = rng.uniform(0.01, 0.02, size=num_peaks)[:, None, None]

        # peak roundness
        xp = rng.randint(1, 3, size=num_peaks)[:, None, None] * 2
        yp = rng.randint(1, 3, size=num_peaks)[:, None, None] * 2

        # evaluate on grid points
        rows = self.height_field_size[0]
        cols = self.height_field_size[1]
        x = np.linspace(-1, 1, rows)[None, :, None]
        y = np.linspace(-1, 1, cols)[None, None, :]
        peaks = scale * np.exp(-((x - x0) ** xp / xs + (y - y0) ** yp / ys))

        # Make into one height field
        peaks = np.sum(peaks, axis=0).flatten()

        # Add some ripples
        noise = rng.uniform(-1, 1, size=self.height_field_size)
        # peaks += gaussian_filter(noise, 1).flatten()

        # Make a flat platform in the centre
        rows = self.height_field_size[0]
        cols = self.height_field_size[1]
        midpoints = peaks.reshape(self.height_field_size)[
            int(rows / 2 - 1) : int(rows / 2 + 2), int(cols / 2 - 1) : int(cols / 2 + 2)
        ]
        midpoints[:] = midpoints.mean()

        # bins = np.linspace(peaks.min(), peaks.max(), self.digitize_bins)
        # digitized = bins[np.digitize(peaks, bins) - 1]
        return peaks - peaks.min()

    def generate_step_placements(self, n_steps=40, phi=None):

        terrain_info = np.zeros((n_steps, 6))
        terrain_info[1, 0] = 0.8
        terrain_info[2, 0] = 1.55

        for i in range(2, n_steps - 1):
            next_step_xyz = terrain_info[i]
            bound_checked_index = (i + 1) % n_steps

            base_yaw = terrain_info[i, 3]
            base_phi = phi[bound_checked_index] if phi is not None else 0

            dr = 0.65
            yaw = 5 * DEG2RAD
            dx = dr * np.cos(yaw + base_phi)
            dx = np.sign(dx) * min(max(abs(dx), 0.25 * 2.5), 0.65)
            dy = dr * np.sin(yaw - base_phi)

            matrix = np.array(
                [
                    [np.cos(base_yaw), -np.sin(base_yaw)],
                    [np.sin(base_yaw), np.cos(base_yaw)],
                ]
            )
            dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

            x = next_step_xyz[0] + dxy[0]
            y = next_step_xyz[1] + dxy[1]
            length_per_index = self.total_length / 128
            x_index = int(x / length_per_index) + 128
            y_index = int(y / length_per_index) + 128
            z = self.data[y_index, x_index] * self.height_scale
            y_tilt = np.arctan2(
                (self.data[y_index + 1, x_index] - self.data[y_index - 1, x_index])
                * self.height_scale,
                length_per_index * 2,
            )
            x_tilt = np.arctan2(
                (self.data[y_index, x_index + 1] - self.data[y_index, x_index - 1])
                * self.height_scale,
                length_per_index * 2,
            )

            matrix = np.array(
                [
                    [-np.cos(base_yaw + yaw), -np.sin(base_yaw + yaw)],
                    [np.sin(base_yaw + yaw), -np.cos(base_yaw + yaw)],
                ]
            )
            x_tilt, y_tilt = np.dot(matrix, np.concatenate(([-y_tilt], [x_tilt])))

            x_tilt = min(x_tilt, 20 * DEG2RAD)
            x_tilt = max(x_tilt, -20 * DEG2RAD)
            y_tilt = min(y_tilt, 20 * DEG2RAD)
            y_tilt = max(y_tilt, -20 * DEG2RAD)

            terrain_info[bound_checked_index, 0] = x
            terrain_info[bound_checked_index, 1] = y
            terrain_info[bound_checked_index, 2] = z
            terrain_info[bound_checked_index, 3] = yaw + base_yaw
            terrain_info[bound_checked_index, 4] = x_tilt
            terrain_info[bound_checked_index, 5] = y_tilt

        return terrain_info

    def set_p_noise_height_field(self):
        import noise

        hfield = np.zeros(self.height_field_size)
        rows, columns = self.height_field_size

        scale = 30.0
        octaves = 6
        persistence = 0.55
        lacunarity = 0.65
        for i in range(rows):
            for j in range(columns):
                hfield[i][j] = noise.pnoise3(
                    (max(126 - i, 129 - i)) / scale,
                    (max(120 - j, 140 - j)) / scale,
                    0 / 100,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=rows,
                    repeaty=columns,
                    base=0,
                )

        height_scale = 5
        hfield[126:129, 125:142] = 0
        hfield_data = hfield.reshape(np.size(hfield)) * height_scale

        hfield_data = hfield_data - hfield_data.min()
        height = hfield_data.max() / 2

        if self.id >= 0:
            self.set_position(pos=(-100000, -100000, -100000))
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[
                    self.total_length / rows * 2,
                    self.total_length / columns * 2,
                    1,
                ],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=rows,
                numHeightfieldColumns=columns,
                # replaceHeightfieldIndex=self.shape_id,
            )
        else:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[
                    self.total_length / rows * 2,
                    self.total_length / columns * 2,
                    1,
                ],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=rows,
                numHeightfieldColumns=columns,
            )
            self.id = self._p.createMultiBody(0, self.shape_id, -1, (0, 0, 0))

            self._p.changeDynamics(
                self.id,
                -1,
                lateralFriction=1.0,
                restitution=0.1,
                contactStiffness=30000,
                contactDamping=1000,
            )

        self._p.changeVisualShape(
            self.id,
            -1,
            textureUniqueId=self.texture_id,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0],
        )

        # self._p.setCollisionFilterGroupMask(self.id, -1, 0, 0)
        self.set_position(pos=(0, 0, height - 1.75 - 0.405))

        self.data = hfield
        self.height_scale = height_scale

        # return self.generate_step_placements()
