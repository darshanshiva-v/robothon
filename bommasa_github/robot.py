import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# IDs
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

bottle_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_free"
)
bottle_qpos_addr = model.jnt_qposadr[bottle_joint_id]

water_geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "water"
)

start = time.time()

state = 0
attached = False
cup_level = 0.0
bottle_level = 1.0

# original bottle position
bottle_home = np.array([0.2, 0, 0.2])

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        t = time.time() - start

        # MOVE TO BOTTLE
        if state == 0:
            target = np.array([0.2, 0.5, -0.3])
            data.qpos[:3] += 0.05 * (target - data.qpos[:3])

            data.ctrl[0] = 0.04
            data.ctrl[1] = 0.04

            if t > 2:
                state = 1

        # GRIP
        elif state == 1:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            attached = True

            if t > 4:
                state = 2

        # MOVE TO GLASS
        elif state == 2:
            target = np.array([0.4, 0.6, -0.2])
            data.qpos[:3] += 0.05 * (target - data.qpos[:3])

            if t > 6:
                state = 3

        # POUR
        elif state == 3:
            data.qpos[2] += 0.05 * (-1.2 - data.qpos[2])

            # simulate water transfer
            if cup_level < 0.5:
                cup_level += 0.003
                bottle_level -= 0.003

            else:
                state = 4

        # RETURN BOTTLE
        elif state == 4:
            data.qpos[2] += 0.05 * (0 - data.qpos[2])

            target = np.array([0.2, 0.5, -0.3])
            data.qpos[:3] += 0.05 * (target - data.qpos[:3])

            if t > 12:
                state = 5

        # RELEASE
        elif state == 5:
            data.ctrl[0] = 0.04
            data.ctrl[1] = 0.04
            attached = False

        # ATTACH BOTTLE
        if attached:
            current = data.qpos[bottle_qpos_addr : bottle_qpos_addr + 3]
            target = data.xpos[gripper_id]
            data.qpos[bottle_qpos_addr : bottle_qpos_addr + 3] = (
                current + 0.2 * (target - current)
            )

        else:
            # place bottle back
            data.qpos[bottle_qpos_addr : bottle_qpos_addr + 3] = bottle_home

        # UPDATE WATER LEVEL IN GLASS
        model.geom_size[water_geom_id][1] = cup_level * 0.1

        mujoco.mj_step(model, data)
        viewer.sync()