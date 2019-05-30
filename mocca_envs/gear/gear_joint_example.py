import pybullet as p
import pybullet_data
import numpy as np
import time
from mocca_envs.bullet_objects import Rectangle
from mocca_envs.bullet_utils import BulletClient


bc = BulletClient(connection_mode=p.GUI)

bc.setGravity(0, 0, -10)

bc.setAdditionalSearchPath("./gear/")
box_id = bc.loadURDF(
    "limbs.urdf", basePosition=[0, 0, 1], baseOrientation=[0, 1.9, 0, 1]
)

c = bc.createConstraint(
    box_id,
    0,
    box_id,
    1,
    jointType=bc.JOINT_GEAR,
    jointAxis=[0, 1, 0],
    # parentFramePosition=[-1, 0, 0],
    # childFramePosition=[-1, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0],
)
bc.changeConstraint(c, gearRatio=5, maxForce=10000)

for i in range(2):
    bc.setJointMotorControl2(
        box_id,
        i,
        controlMode=bc.POSITION_CONTROL,
        targetPosition=0,
        targetVelocity=0,
        positionGain=0.1,
        velocityGain=0.1,
        force=0,
    )


bc.resetJointState(box_id, 0, targetValue=-0.1)
bc.resetDebugVisualizerCamera(1, 0, 0, [0, 0, 1])
bc.configureDebugVisualizer(bc.COV_ENABLE_GUI, 0)


for i in range(2000):
    p.stepSimulation()
    time.sleep(0.01)
    if i < 10:
        bc.setJointMotorControl2(
            bodyIndex=box_id, jointIndex=1, controlMode=bc.TORQUE_CONTROL, force=150
        )
