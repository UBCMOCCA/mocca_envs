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
    id="CassieMocapEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMocapEnv",
    max_episode_steps=1000,
)

register(
    id="CassieMocapPhaseRSIEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMocapPhaseEnv",
    max_episode_steps=1000,
    kwargs={"rsi": True},
)

register(
    id="Cassie2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassieMocap2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMocapEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="Walker3DCustomEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DChairEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DChairEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DTerrainEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DTerrainEnv",
    max_episode_steps=1000,
)
