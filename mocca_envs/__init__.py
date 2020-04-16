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
    id="CassieOSUEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieOSUEnv",
    max_episode_steps=1000,
)

register(
    id="CassieOSU2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieOSUEnv",
    max_episode_steps=1000,
    kwargs={"planar": True},
)

register(
    id="CassieRSIEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieOSUEnv",
    max_episode_steps=1000,
    kwargs={"residual_control": False},
)

register(
    id="CassieRSI2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieOSUEnv",
    max_episode_steps=1000,
    kwargs={"planar": True, "residual_control": False},
)

register(
    id="CassieMirror2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMirrorEnv",
    max_episode_steps=300,
    kwargs={"planar": True},
)

register(
    id="CassieMirrorPhase2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMirrorEnv",
    max_episode_steps=300,
    kwargs={"planar": True, "phase_in_obs": True},
)

register(
    id="CassiePhaseEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMirrorEnv",
    max_episode_steps=1000,
    kwargs={"phase_in_obs": True, "residual_control": True, "rsi":True},
)

register(
    id="CassiePhase2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMirrorEnv",
    max_episode_steps=1000,
    kwargs={"phase_in_obs": True, "residual_control": True, "planar": True},
)

register(
    id="CassieMirrorEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieMirrorEnv",
    max_episode_steps=300,
    kwargs={},
)

register(
    id="CassieNoMocap2DEnv-v0",
    entry_point="mocca_envs.cassie_envs:CassieOSUEnv",
    max_episode_steps=1000,
    kwargs={"planar": True, "residual_control": False, "rsi": True},
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
    id="MikeStepperEnv-v0",
    entry_point="mocca_envs.mike:MikeStepperEnv",
    max_episode_steps=1000,
)

register(
    id="Walker2DCustomEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker2DCustomEnv",
    max_episode_steps=1000,
)

register(
    id="Walker2DStepperEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker2DStepperEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DMocapEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DMocapEnv",
    max_episode_steps=1000,
)

register(
    id="Walker3DMocapStepperEnv-v0",
    entry_point="mocca_envs.walker3d_envs:Walker3DMocapStepperEnv",
    max_episode_steps=1000,
)
