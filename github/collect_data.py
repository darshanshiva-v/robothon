import mujoco
import mujoco.viewer
import cv2
import time
import os

model = mujoco.MjModel.from_xml_path("mujoco/scene.xml")
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, width=640, height=480)

os.makedirs("dataset", exist_ok=True)

count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        renderer.update_scene(data)
        frame = renderer.render()

        cv2.imshow("camera", frame)
        cv2.waitKey(1)

        # Save images
        if count % 10 == 0:
            filename = f"dataset/img_{int(time.time()*1000)}.png"
            cv2.imwrite(filename, frame)
            print("Saved:", filename)

        count += 1

        mujoco.mj_step(model, data)
        viewer.sync()
