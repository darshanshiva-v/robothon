import warnings
warnings.filterwarnings("ignore")

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from ultralytics import YOLO

# =========================
# ✅ LOAD YOLO MODEL
# =========================
yolo_model = YOLO("/home/pragathi/physical-ai-challange-2026/runs/detect/train3/weights/best.pt")

# =========================
# ✅ LOAD MUJOCO MODEL
# =========================
model = mujoco.MjModel.from_xml_path("mujoco/scene.xml")
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model)

target = np.array([0.0, -0.9, 1.1, 0.45, 0.0, 0.35], dtype=float)

# =========================
# ✅ MAIN LOOP
# =========================
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        success = False

        while viewer.is_running():

            mujoco.mj_step(model, data)
            viewer.sync()

            # CAMERA VIEW FROM THE REAL SO-101 WRIST/CAMERA MOUNT
            renderer.update_scene(data, camera="arm_cam")
            frame = renderer.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # YOLO DETECTION
            results = yolo_model(frame)

            centers = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    centers.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # FORCE TWO OBJECTS
            if len(centers) >= 2:
                plug_center = centers[0]
                socket_center = centers[1]

                cv2.circle(frame, plug_center, 5, (0,0,255), -1)
                cv2.circle(frame, socket_center, 5, (255,0,0), -1)
                cv2.line(frame, plug_center, socket_center, (0,255,0), 2)

                image_dx = socket_center[0] - plug_center[0]
                image_dy = socket_center[1] - plug_center[1]

                # Drive the actual SO-101 actuators with a simple visual-servo rule.
                target[0] = np.clip(target[0] + 0.0008 * image_dx, -1.5, 1.5)
                target[1] = np.clip(target[1] - 0.0006 * image_dy, -1.5, 0.2)
                target[2] = np.clip(target[2] + 0.0006 * image_dy, -0.5, 1.5)
                target[5] = 0.15

                distance = np.linalg.norm(
                    np.array(plug_center) - np.array(socket_center)
                )

                print("Distance:", distance)

                # SUCCESS CONDITION
                if distance < 10 and not success:
                    print("✅ INSERT SUCCESS (simulated)")
                    target[1] = -1.2
                    target[2] = 1.2
                    target[5] = 0.0
                    success = True

            data.ctrl[:] = target

            # SHOW CAMERA
            cv2.imshow("Arm Camera + AI", frame)

            # AUTO EXIT AFTER SUCCESS
            if success:
                cv2.waitKey(2000)
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

except Exception as e:
    print("Stopped safely:", e)

# CLEAN EXIT
try:
    cv2.destroyAllWindows()
except:
    pass
