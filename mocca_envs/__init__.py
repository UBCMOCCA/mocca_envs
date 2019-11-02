import gym
import os


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


# fixing package path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


register(
    id="CassieEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieEnv",
    max_episode_steps=1000,
)

register(
    id="Cassie2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassiePhaseMocca2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassiePhaseMoccaEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassiePhaseMirror2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassiePhaseMirrorEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="Child3DCustomEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Child3DCustomEnv",
    max_episode_steps=1000,
)


register(
    id="Walker3DCustomEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DStepperEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DStepperEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DChairEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DChairEnv",
    max_episode_steps=1000,
)
