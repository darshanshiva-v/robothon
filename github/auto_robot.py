import cv2
from ultralytics import YOLO
import numpy as np

# Load trained model
model = YOLO("/home/pragathi/physical-ai-challange-2026/runs/detect/train3/weights/best.pt")

# Fake robot position (simulation)
robot_x, robot_y = 0, 0

def move_robot_towards(target_x, target_y):
    global robot_x, robot_y

    dx = target_x - robot_x
    dy = target_y - robot_y

    robot_x += dx * 0.1
    robot_y += dy * 0.1

    print(f"Moving robot → ({robot_x:.2f}, {robot_y:.2f})")

# Load one image (simulation)
img_path = "dataset/images/img_1776163818381.png"
frame = cv2.imread(img_path)

while True:
    results = model(frame)

    plug_center = None
    socket_center = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls == 0:  # plug
                plug_center = (cx, cy)
                cv2.circle(frame, plug_center, 5, (0, 0, 255), -1)

            elif cls == 1:  # socket
                socket_center = (cx, cy)
                cv2.circle(frame, socket_center, 5, (255, 0, 0), -1)

    if plug_center and socket_center:
        cv2.line(frame, plug_center, socket_center, (0, 255, 0), 2)

        print("Plug:", plug_center)
        print("Socket:", socket_center)

        # 🔥 MOVE ROBOT
        move_robot_towards(socket_center[0], socket_center[1])

        # 🔥 FIXED DISTANCE CALCULATION
        distance = np.linalg.norm(np.array([robot_x, robot_y]) - np.array(socket_center))

        print("Distance:", distance)

        # 🔥 FIXED CONDITION
        if distance < 10:
            print("✅ INSERT SUCCESS (simulated)")
            break

    cv2.imshow("Robot Simulation", frame)

    if cv2.waitKey(1000) == 27:
        break

cv2.destroyAllWindows()
