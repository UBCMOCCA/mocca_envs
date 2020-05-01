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
        self.n_steps = 100

        texture_file = os.path.join(current_dir, "data", "misc", "checker_blue2.png")
        self.texture_id = self._p.loadTexture(texture_file)
        self.texture_scaling = 10

        self.digitize_bins = 32

        self.base_phi = DEG2RAD * np.array(
            [-10] + [20, -20] * (self.n_steps // 2 - 1) + [10]
        )

        self.total_length = 25.6

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
        height = self.data.max() / 2 - self.data[midpoint: midpoint + 1].mean()
        self.data -= self.data[midpoint: midpoint + 1].mean()

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
                    int(rows / 2 - 1): int(rows / 2 + 2), int(cols / 2 - 1): int(cols / 2 + 2)
                    ]
        midpoints[:] = midpoints.mean()

        # bins = np.linspace(peaks.min(), peaks.max(), self.digitize_bins)
        # digitized = bins[np.digitize(peaks, bins) - 1]
        return peaks - peaks.min()

    def generate_step_placements(self, hfield, height_scale):
        n_steps = self.n_steps
        DEG2RAD = np.pi / 180
        y_range = np.array([-10, -10]) * DEG2RAD
        p_range = np.array([90 - 0, 90 + 0]) * DEG2RAD
        t_range = np.array([-0, 0]) * DEG2RAD

        dr = np.random.uniform(0.65, 0.65, size=n_steps)
        dphi = np.random.uniform(*y_range, size=n_steps)
        dtheta = np.random.uniform(*p_range, size=n_steps)
        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        # make first step slightly further to accommodate different starting poses
        dr[1] = 0.8
        dphi[1] = 0.0
        dtheta[1] = np.pi / 2

        dr[2] = 0.75
        dphi[2] = 0.0
        dtheta[2] = np.pi / 2

        x_tilt = np.random.uniform(*t_range, size=n_steps)
        y_tilt = np.random.uniform(*t_range, size=n_steps)
        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi_copy = np.copy(dphi)
        dphi = np.cumsum(dphi)

        x_ = dr * np.sin(dtheta) * np.cos(dphi)
        y_ = dr * np.sin(dtheta) * np.sin(dphi)
        z_ = dr * np.cos(dtheta)

        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_) + 0

        terrain_info = np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

        for i in range(2, n_steps - 1):
            next_step_xyz = terrain_info[i]
            bound_checked_index = (i + 1) % n_steps

            base_yaw = terrain_info[i, 3]
            base_phi = self.base_phi[bound_checked_index]

            dr = np.random.uniform(0.65, 0.65, 1)
            yaw = 5 * DEG2RAD
            # if yaw + base_yaw >= np.pi/2 + 0.2:
            #     yaw = np.pi/2+0.2-base_yaw
            dx = dr * np.cos(yaw + base_phi)
            dy = dr * np.sin(yaw - base_phi)

            matrix = np.array([
                [np.cos(base_yaw), -np.sin(base_yaw)],
                [np.sin(base_yaw), np.cos(base_yaw)]
            ])
            dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

            x = next_step_xyz[0] + dxy[0]
            y = next_step_xyz[1] + dxy[1]
            length_per_index = (self.total_length / 128)
            x_index = int(x / length_per_index) + 128
            y_index = int(y / length_per_index) + 128
            z = hfield[y_index, x_index] * height_scale
            y_tilt = np.arctan2((hfield[y_index + 1, x_index] - hfield[y_index - 1, x_index]) * height_scale,
                                length_per_index * 2)
            x_tilt = np.arctan2((hfield[y_index, x_index + 1] - hfield[y_index, x_index - 1]) * height_scale,
                                length_per_index * 2)
            print("tilt", x_tilt, y_tilt)

            matrix = np.array([
                [-np.cos(base_yaw + yaw), -np.sin(base_yaw + yaw)],
                [np.sin(base_yaw + yaw), -np.cos(base_yaw + yaw)]
            ])
            x_tilt, y_tilt = np.dot(matrix, np.concatenate(([-y_tilt], [x_tilt])))

            # x_tilt = min(x_tilt, 20*DEG2RAD)
            # x_tilt = max(x_tilt, -20*DEG2RAD)
            # y_tilt = min(y_tilt, 20*DEG2RAD)
            # y_tilt = max(y_tilt, -20*DEG2RAD)

            terrain_info[bound_checked_index, 0] = x
            terrain_info[bound_checked_index, 1] = y
            terrain_info[bound_checked_index, 2] = z
            terrain_info[bound_checked_index, 3] = yaw + base_yaw
            terrain_info[bound_checked_index, 4] = x_tilt
            terrain_info[bound_checked_index, 5] = y_tilt

        return terrain_info

    def get_p_noise_height_field(self):
        import noise
        hfield = np.zeros((256, 256))
        scale = 30.0
        octaves = 6
        persistence = 0.55
        lacunarity = 0.65
        for i in range(256):
            for j in range(256):
                z = np.random.randn(1)
                # print(z)
                hfield[i][j] = noise.pnoise3((max(126 - i, 129 - i)) / scale,
                                             (max(120 - j, 140 - j)) / scale, 0 / 100,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             repeatx=256,
                                             repeaty=256,
                                             base=0)
                if i > 120 and i < 140:
                    print(i, j, hfield[i][j])
                # hfield[i][j] = max(0.01, hfield[i][j])
        # for i in range(256):
        #     for j in range(256):
        #         hfield[i][j] = 1+np.cos((i - 130)/20.0*np.pi*2)
        # hfield -= hfield.min()
        height_scale = 5
        hfield[126:129, 125:140] = 0
        # hfield -= hfield.min()
        hfield_data = hfield.reshape(256 * 256) * height_scale

        hfield_data = hfield_data - hfield_data.min()
        height = hfield_data.max() / 2
        hfield_data = hfield_data

        if self.id >= 0:
            self.set_position(pos=(-100000, -100000, -100000))
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[self.total_length / 128, self.total_length / 128, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
                # replaceHeightfieldIndex=self.shape_id,
            )
        else:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[self.total_length / 128, self.total_length / 128, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
            )
            self.id = self._p.createMultiBody(0, self.shape_id, -1, (0, 0, 0))

            self._p.changeDynamics(self.id, -1, lateralFriction=1.0,
                                   restitution=0.1,
                                   contactStiffness=30000,
                                   contactDamping=1000)

        self._p.changeVisualShape(
            self.id,
            -1,
            textureUniqueId=self.texture_id,
            # rgbaColor=[0.65, 0.37, 0.04, 1],
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0],
        )

        # self._p.setCollisionFilterGroupMask(self.id, -1, 0, 0)

        self.set_position(pos=(0, 0, height - 1.75 - 0.408))
        # print(hfield_data.min(), hfield_data.max(), height-1.3)
        return self.generate_step_placements(hfield, height_scale)
        # print(z_min, z_max, terrain_info[:, 2])

    def generate_height_field_from_step(self, terrain_info):
        num_steps, _ = terrain_info.shape
        height_field_col = self.height_field_size[1]
        height_field_row = self.height_field_size[0]
        # height_field_size = np.copy(self.sim.model.hfield_size)

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
        x_scale = (x_max - x_min + plank_radius * 2) / 2
        y_scale = 25.6
        x_pos = (x_max + x_min) / 2
        z_pos = z_min

        from scipy import interpolate
        u = np.copy(terrain_info[:, 0])
        tick, u = interpolate.splprep(terrain_info[:, 2:3].T, u=u, k=3, s=0)
        u = np.linspace(x_min - plank_radius, x_max + plank_radius, 256)
        out = interpolate.splev(u, tick)

        height_field_data[:, :] = 0
        length_per_index = x_scale * 2 / 256
        width_per_index = y_scale * 2 / 256
        for i in range(256):
            y_indices = 127
            h_data = out[0][i]  # (out[0][i]-z_min)/(z_max-z_min)
            height_field_data[y_indices - 3:y_indices + 3, i] = h_data  # (out[0][i]-z_min)/(z_max-z_min)#*z_scale
            # print((out[0][i]-z_min)/(z_max-z_min)*z_scale)

        hfield_data = height_field_data.reshape(height_field_col * height_field_row)
        z_pos = z_min

        if self.id >= 0:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[x_scale / 128, y_scale / 128, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
                replaceHeightfieldIndex=self.shape_id,
            )
        else:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[x_scale / 128, y_scale / 128, 1],
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

    def generate_height_field_from_step_2d(self, terrain_info):
        num_steps, _ = terrain_info.shape
        height_field_col = self.height_field_size[1]
        height_field_row = self.height_field_size[0]
        # height_field_size = np.copy(self.sim.model.hfield_size)

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
        x_scale = (x_max - x_min + 5 * 2) / 2
        y_min = min(terrain_info[:, 1])
        y_max = max(terrain_info[:, 1])
        y_scale = 12.8
        x_pos = (x_max + x_min) / 2
        y_pos = 0  # (y_max+y_min) / 2
        z_pos = z_min

        height_field_data[:, :] = 0

        for displacement in np.linspace(-0.4, 0.4, 160):
            from scipy import interpolate
            u = np.arange(num_steps)
            terrain_info_translate = np.copy(terrain_info)
            terrain_info_translate[:, 0] -= displacement * np.sin(terrain_info[:, 3])
            terrain_info_translate[:, 1] += displacement * np.cos(terrain_info[:, 3])
            tick, u = interpolate.splprep(terrain_info_translate[:, 0:3].T, u=u, k=3, s=0)
            u = np.linspace(0, num_steps - 1, 256)
            out = interpolate.splev(u, tick)

            length_per_index = x_scale * 2 / 256
            width_per_index = y_scale * 2 / 256
            for i in range(255):
                x_indices = (out[0][i] - x_pos) / length_per_index
                x_next_indices = (out[0][i + 1] - x_pos) / length_per_index
                x_indices = int(x_indices + 127)
                x_next_indices = int(x_next_indices + 127)
                if x_indices > x_next_indices:
                    temp = x_indices
                    x_indices = x_next_indices
                    x_next_indices = temp

                y_indices = (out[1][i]) / width_per_index
                y_next_indices = (out[1][i + 1]) / width_per_index
                y_indices = int(y_indices + 127)
                y_next_indices = int(y_next_indices + 127)
                if y_indices > y_next_indices:
                    temp = y_indices
                    y_indices = y_next_indices
                    y_next_indices = temp
                # y_indices = int((out[1][i] - 0)/width_per_index+127)
                height_field_data[y_indices, x_indices:x_next_indices + 1] = out[2][i]
                print(x_indices, x_next_indices)

        hfield_data = height_field_data.reshape(height_field_col * height_field_row)
        z_pos = z_min

        if self.id >= 0:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[x_scale / 128, y_scale / 128, 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=256,
                numHeightfieldColumns=256,
                replaceHeightfieldIndex=self.shape_id,
            )
        else:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[x_scale / 128, y_scale / 128, 1],
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
            # textureUniqueId=self.texture_id,
            rgbaColor=[0.65, 0.37, 0.04, 1],
            specularColor=[0.65, 0.37, 0.04],
        )

        self._p.setCollisionFilterGroupMask(self.id, -1, 0, 0)

        self.set_position(pos=(x_pos, 0, 2.5))
        print(z_min, z_max, terrain_info[:, 2])
