import argparse
import os
from dataclasses import dataclass

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


MODEL_PATH = "mujoco/scene.xml"
POLICY_PATH = "models/so101_pick_place_policy.pt"
VISION_CAMERA = "overview"
DISPLAY_CAMERA = "arm_cam"
CONTROL_DT = 0.01

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

POSES = {
    "home": np.array([0.0, -0.75, 1.05, 0.55, 0.0, 0.75], dtype=float),
    "approach_pick": np.array([-0.68, -1.02, 1.15, 0.62, 0.0, 0.72], dtype=float),
    "pick": np.array([-0.68, -1.18, 1.24, 0.70, 0.0, 0.72], dtype=float),
    "grasp": np.array([-0.68, -1.18, 1.24, 0.70, 0.0, 0.05], dtype=float),
    "lift": np.array([-0.68, -0.88, 0.98, 0.56, 0.0, 0.05], dtype=float),
    "approach_place": np.array([0.68, -1.00, 1.12, 0.62, 0.0, 0.05], dtype=float),
    "place": np.array([0.68, -1.14, 1.20, 0.68, 0.0, 0.05], dtype=float),
    "release": np.array([0.68, -1.14, 1.20, 0.68, 0.0, 0.75], dtype=float),
    "retreat": np.array([0.68, -0.86, 0.96, 0.52, 0.0, 0.75], dtype=float),
}

EXPERT_PLAN = [
    ("home", 0.7),
    ("approach_pick", 1.8),
    ("pick", 1.2),
    ("grasp", 1.0),
    ("lift", 1.2),
    ("approach_place", 1.9),
    ("place", 1.1),
    ("release", 0.9),
    ("retreat", 1.0),
    ("home", 1.2),
]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class VisionResult:
    red_center: np.ndarray
    blue_center: np.ndarray
    red_visible: float
    blue_visible: float
    frame_bgr: np.ndarray


class SO101PickPlaceDemo:
    def __init__(self, controller_mode: str, policy_path: str):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)

        self.site_ids = {
            "gripperframe": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
            "pick_site": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pick_site"),
            "place_site": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "place_site"),
        }
        block_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
        self.block_qpos_adr = self.model.jnt_qposadr[block_joint_id]
        self.block_qvel_adr = self.model.jnt_dofadr[block_joint_id]

        self.controller_mode = controller_mode
        self.policy_path = policy_path
        self.policy = None
        self.obs_dim = 19
        if controller_mode == "ml":
            self.policy = self._load_policy(policy_path)

        self.target_ctrl = POSES["home"].copy()
        self.stage_index = 0
        self.stage_elapsed = 0.0
        self.attached = False
        self.released = False
        self.cycle_complete = False
        self.grasp_offset = np.array([0.0, 0.0, -0.02], dtype=float)
        self.reset()

    def _load_policy(self, policy_path: str):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"Missing learned policy at {policy_path}. Run: python3 train_policy.py"
            )
        checkpoint = torch.load(policy_path, map_location="cpu")
        policy = PolicyNet(checkpoint["obs_dim"], checkpoint["act_dim"])
        policy.load_state_dict(checkpoint["state_dict"])
        policy.eval()
        return policy

    def site_position(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.site_ids[name]].copy()

    def set_block_pose(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.data.qpos[self.block_qpos_adr:self.block_qpos_adr + 3] = pos
        self.data.qpos[self.block_qpos_adr + 3:self.block_qpos_adr + 7] = quat
        self.data.qvel[self.block_qvel_adr:self.block_qvel_adr + 6] = 0.0

    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = POSES["home"]
        self.target_ctrl = POSES["home"].copy()
        self.attached = False
        self.released = False
        self.cycle_complete = False
        self.stage_index = 0
        self.stage_elapsed = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.set_block_pose(self.site_position("pick_site") + np.array([0.0, 0.0, -0.018]))
        mujoco.mj_forward(self.model, self.data)

    def detect_scene(self) -> VisionResult:
        self.renderer.update_scene(self.data, camera=VISION_CAMERA)
        frame_rgb = self.renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([12, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 70]), np.array([180, 255, 255]))
        red_mask = red_mask1 + red_mask2
        blue_mask = cv2.inRange(hsv, np.array([95, 90, 60]), np.array([130, 255, 255]))

        red_center, red_visible = self._mask_center(red_mask)
        blue_center, blue_visible = self._mask_center(blue_mask)
        return VisionResult(red_center, blue_center, red_visible, blue_visible, frame_bgr)

    @staticmethod
    def _mask_center(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(2, dtype=float), 0.0
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            return np.zeros(2, dtype=float), 0.0
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            return np.zeros(2, dtype=float), 0.0
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        h, w = mask.shape
        return np.array([cx / w, cy / h], dtype=float), 1.0

    def observation(self, vision: VisionResult) -> np.ndarray:
        qpos = self.data.qpos[:6].astype(np.float32)
        ctrl = self.data.ctrl[:6].astype(np.float32)
        stage_phase = self.stage_index / max(len(EXPERT_PLAN) - 1, 1)
        obs = np.concatenate([
            qpos,
            ctrl,
            vision.red_center.astype(np.float32),
            vision.blue_center.astype(np.float32),
            np.array([
                vision.red_visible,
                vision.blue_visible,
                float(self.attached),
            ], dtype=np.float32),
        ])
        return obs

    def maybe_attach_or_release(self):
        grip = self.site_position("gripperframe")
        block = self.data.qpos[self.block_qpos_adr:self.block_qpos_adr + 3].copy()
        dist = np.linalg.norm(grip - block)
        gripper_closed = self.data.ctrl[5] < 0.12
        at_pick_stage = self.stage_index >= 3 and self.stage_index <= 5
        at_place_stage = self.stage_index >= 6

        if (not self.attached) and gripper_closed and dist < 0.05 and at_pick_stage:
            self.attached = True
            self.released = False

        if self.attached and self.data.ctrl[5] > 0.60 and at_place_stage:
            self.attached = False
            self.released = True

    def apply_object_physics_assist(self):
        if self.attached:
            self.set_block_pose(self.site_position("gripperframe") + self.grasp_offset)
        elif self.released:
            place = self.site_position("place_site") + np.array([0.0, 0.0, -0.018])
            block = self.data.qpos[self.block_qpos_adr:self.block_qpos_adr + 3].copy()
            self.set_block_pose(block + 0.15 * (place - block))

    def expert_action(self, vision: VisionResult) -> np.ndarray:
        stage_name, duration = EXPERT_PLAN[self.stage_index]
        pose = POSES[stage_name].copy()

        if vision.red_visible:
            red_x = vision.red_center[0] - 0.5
            pose[0] += np.clip(-0.7 * red_x, -0.12, 0.12)
        if self.stage_index >= 5 and vision.blue_visible:
            blue_x = vision.blue_center[0] - 0.5
            pose[0] += np.clip(-0.7 * blue_x, -0.12, 0.12)

        if self.stage_elapsed >= duration:
            self.stage_index += 1
            self.stage_elapsed = 0.0
            if self.stage_index >= len(EXPERT_PLAN):
                self.stage_index = len(EXPERT_PLAN) - 1
                self.cycle_complete = True

        return pose

    def ml_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            out = self.policy(x).squeeze(0).cpu().numpy()
        return out

    def step(self):
        vision = self.detect_scene()
        obs = self.observation(vision)

        if self.controller_mode == "expert":
            action = self.expert_action(vision)
        else:
            action = self.ml_action(obs)

        self.target_ctrl = np.clip(
            action.astype(float),
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        self.data.ctrl[:] = self.data.ctrl + 0.08 * (self.target_ctrl - self.data.ctrl)
        self.maybe_attach_or_release()

        for _ in range(4):
            self.apply_object_physics_assist()
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.stage_elapsed += self.model.opt.timestep

        return vision

    def overlay(self, display_frame: np.ndarray, vision: VisionResult):
        if vision.red_visible:
            h, w, _ = display_frame.shape
            pt = (int(vision.red_center[0] * w), int(vision.red_center[1] * h))
            cv2.circle(display_frame, pt, 8, (0, 0, 255), -1)
        if vision.blue_visible:
            h, w, _ = display_frame.shape
            pt = (int(vision.blue_center[0] * w), int(vision.blue_center[1] * h))
            cv2.circle(display_frame, pt, 8, (255, 0, 0), -1)

        label = f"mode={self.controller_mode} stage={self.stage_index + 1}/{len(EXPERT_PLAN)} attached={self.attached}"
        cv2.putText(display_frame, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 220, 30), 2)
        cv2.putText(display_frame, "SO-101 autonomous box-to-box pick and place", (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self.cycle_complete:
            cv2.putText(display_frame, "Cycle complete", (14, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (80, 220, 255), 2)

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 1.4
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = [0.0, 0.0, 0.18]

            while viewer.is_running():
                vision = self.step()
                self.renderer.update_scene(self.data, camera=DISPLAY_CAMERA)
                frame_rgb = self.renderer.render()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self.overlay(frame_bgr, vision)
                cv2.imshow("SO-101 Camera", frame_bgr)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                viewer.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["expert", "ml"], default="expert")
    parser.add_argument("--policy-path", default=POLICY_PATH)
    args = parser.parse_args()

    demo = SO101PickPlaceDemo(controller_mode=args.controller, policy_path=args.policy_path)
    demo.run()


if __name__ == "__main__":
    main()
