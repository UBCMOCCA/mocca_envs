import gym


def register(id, **kvargs):
    if id in gym.envs.registration.registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, **kvargs)


register(
    id="CassieEnv-v0", entry_point="mocca_envs.envs:CassieEnv", max_episode_steps=1000
)

register(
    id="Walker3DCustomEnv-v0",
    entry_point="mocca_envs.envs:Walker3DCustomEnv",
    max_episode_steps=1000,
)
