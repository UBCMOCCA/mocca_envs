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

        texture_file = os.path.join(current_dir, "data", "misc", "canyon.jpg")
        self.texture_id = self._p.loadTexture(texture_file)
        self.texture_scaling = 1

        self.digitize_bins = 32

    def set_position(self, pos=None, quat=None):
        pos = np.zeros(3) if pos is None else pos
        quat = np.array([0, 0, 0, 1]) if quat is None else quat

        self._p.resetBasePositionAndOrientation(
            self.id, posObj=pos, ornObj=quat
        )

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
        #peaks += gaussian_filter(noise, 1).flatten()

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

    def generate_height_field_from_step(self, terrain_info):
        num_steps, _ = terrain_info.shape
        height_field_col = self.height_field_size[1]
        height_field_row = self.height_field_size[0]
        #height_field_size = np.copy(self.sim.model.hfield_size)

        height_field_data = np.zeros((height_field_row, height_field_col))
        height_field_data = height_field_data.reshape(height_field_col, height_field_row)

        z_max = -100
        z_min = 0
        plank_radius = 0.25
        for i in range(num_steps):
            if terrain_info[i, 2] > z_max:
                z_max = terrain_info[i, 2]
        z_scale = z_max - z_min

        x_min = min(terrain_info[:, 0])
        x_max = max(terrain_info[:, 0])
        x_scale = (x_max - x_min + plank_radius*2) / 2
        y_scale = 25.6
        x_pos = (x_max+x_min) / 2
        z_pos = z_min

        from scipy import interpolate
        u = np.copy(terrain_info[:, 0])
        tick, u = interpolate.splprep(terrain_info[:, 2:3].T, u=u, k=3, s=0)
        u = np.linspace(x_min-plank_radius, x_max+plank_radius, 256)
        out = interpolate.splev(u, tick)
        
        height_field_data[:, :] = 0
        length_per_index = x_scale * 2 / 256
        width_per_index = y_scale * 2 / 256
        for i in range(256):
            y_indices = 127
            h_data = out[0][i]#(out[0][i]-z_min)/(z_max-z_min)
            height_field_data[y_indices-3:y_indices+3, i] = h_data#(out[0][i]-z_min)/(z_max-z_min)#*z_scale
            #print((out[0][i]-z_min)/(z_max-z_min)*z_scale)

        hfield_data = height_field_data.reshape(height_field_col*height_field_row)
        z_pos = z_min

        if self.id >= 0:
            self.shape_id = self._p.createCollisionShape(
                    shapeType=self._p.GEOM_HEIGHTFIELD,
                    meshScale=[x_scale/128, y_scale/128, 1],
                    heightfieldTextureScaling=self.texture_scaling,
                    heightfieldData=hfield_data,
                    numHeightfieldRows=256,
                    numHeightfieldColumns=256,
                    replaceHeightfieldIndex=self.shape_id,
                )
        else:
            self.shape_id = self._p.createCollisionShape(
                    shapeType=self._p.GEOM_HEIGHTFIELD,
                    meshScale=[x_scale/128, y_scale/128, 1],
                    heightfieldTextureScaling=self.texture_scaling,
                    heightfieldData=hfield_data,
                    numHeightfieldRows=256,
                    numHeightfieldColumns=256,
                )
            self.id = self._p.createMultiBody(0, self.shape_id, -1, (0, 0, 0))

            self._p.changeDynamics(self.id, -1, lateralFriction=0.7, restitution=0.2)

        self._p.changeVisualShape(
            self.id,
            -1,
            textureUniqueId=self.texture_id,
            rgbaColor=[0.65, 0.37, 0.04, 1],
            specularColor=[0, 0, 0],
        )

        self._p.setCollisionFilterGroupMask(self.id, -1, 0, 0)

        self.set_position(pos=(x_pos, 0, 2.5))
        print(z_min, z_max, terrain_info[:, 2])