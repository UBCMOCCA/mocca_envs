import gym
import gym.utils.seeding
import pybullet

from mocca_envs.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene


class EnvBase(gym.Env):
    def __init__(self, robot_class, render=False):
        self.scene = None
        self.physics_client_id = -1
        self.owns_physics_client = 0
        self.state_id = -1

        self.is_render = render
        self.robot_class = robot_class

        self.seed()
        self.initialize_scene_and_robot()

    def close(self):
        if self.owns_physics_client and self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):

        self.owns_physics_client = True

        bc_mode = pybullet.GUI if self.is_render else pybullet.DIRECT
        self._p = BulletClient(connection_mode=bc_mode)

        if self.is_render:
            self.camera = Camera(self._p, 1 / self.control_step)
            if hasattr(self, "create_target"):
                self.create_target()

        self.physics_client_id = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        self.scene = SinglePlayerStadiumScene(
            self._p,
            gravity=9.8,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )
        self.scene.initialize()

        # Create floor
        self.ground_ids = {(self.scene.ground_plane_mjcf[0], -1)}

        # Create robot object
        self.robot = self.robot_class(self._p)
        self.robot.initialize()
        self.robot.np_random = self.np_random

        # Create terrain
        if hasattr(self, "create_terrain"):
            self.create_terrain()

        self.state_id = self._p.saveState()

    def set_env_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def set_robot_params(self, params_dict):
        for k, v in params_dict.items():
            if hasattr(self.robot, k):
                setattr(self.robot, k, v)

        # Right now only power can be set
        # Make sure to recalculate torque limit
        self.robot.calc_torque_limits()

    def render(self, mode="human"):
        # Taken care of by pybullet
        if not self.is_render:
            self.is_render = True
            self._p.disconnect()
            self.initialize_scene_and_robot()
            self.reset()

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self):
        keys = self._p.getKeyboardEvents()
        # keys is a dict, so need to check key exists
        if ord("d") in keys and keys[ord("d")] == self._p.KEY_WAS_RELEASED:
            self.debug = True if not hasattr(self, "debug") else not self.debug
        elif ord("r") in keys and keys[ord("r")] == self._p.KEY_WAS_RELEASED:
            self.done = True
        elif ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if ord("z") in keys and keys[ord("z")] == self._p.KEY_WAS_RELEASED:
                    break
