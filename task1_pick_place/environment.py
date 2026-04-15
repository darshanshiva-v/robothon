#!/usr/bin/env python3
"""
environment.py — Shared MuJoCo pick-and-place environment for Task 1.

Canonical contract:
  - Observation: 13-dim float32
      [joint_angles(7), cube_pos(3), target_pos(3)]
  - Action: 6-dim float32
      [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
  - reset(...): returns observation only
  - step(action): returns (obs, reward, done, info)

All Task 1 scripts should treat this file as the source of truth for reset
randomization, episode length, camera names, rendering, and grasp behavior.
"""

import os
import numpy as np
import mujoco


_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MUJOCO_DIR = os.path.join(
    _REPO, "workshop", "dev", "docker", "workspace", "src", "so101_mujoco", "mujoco"
)
_SCENE_XML = os.path.join(_MUJOCO_DIR, "scene.xml")


HOME_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
    "gripper_moving": 0.0,
}


def _joint_adr(model, name, obj_type=mujoco.mjtObj.mjOBJ_JOINT):
    jid = mujoco.mj_name2id(model, obj_type, name)
    if jid < 0:
        return None, None
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


class PickPlaceEnv:
    OBS_DIM = 13
    ACTION_DIM = 6
    EPISODE_LEN = 400
    RENDER_WIDTH = 640
    RENDER_HEIGHT = 480
    TOPDOWN_CAMERA_NAME = "track"
    WRIST_CAMERA_NAME = "d435i"
    CUBE_BODY_NAME = "red_box"
    CUBE_FREE_JOINT_NAME = "red_box_joint"
    GRASP_SITE_NAME = "gripperframe"

    CUBE_START = np.array([0.15, 0.0, 0.025], dtype=np.float32)
    TARGET = np.array([0.30, 0.10, 0.025], dtype=np.float32)
    CUBE_RANDOM_XY_RANGE = np.array([0.025, 0.025], dtype=np.float32)
    TARGET_RANDOM_XY_RANGE = np.array([0.025, 0.025], dtype=np.float32)

    GRASP_CLOSE_THRESHOLD = 0.5
    RELEASE_OPEN_THRESHOLD = 0.3
    GRASP_DISTANCE_THRESHOLD = 0.07
    DEFAULT_CUBE_Z = 0.025

    def __init__(self, scene_xml_path=None):
        self.scene_xml_path = scene_xml_path or _SCENE_XML
        if not os.path.exists(self.scene_xml_path):
            raise FileNotFoundError(f"scene.xml not found at {self.scene_xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.scene_xml_path)
        self.data = mujoco.MjData(self.model)

        self._joint_names = list(HOME_JOINTS.keys())
        self._arm_action_names = self._joint_names[: self.ACTION_DIM]
        self._qpos_adr = {}
        self._qvel_adr = {}
        for name in self._joint_names:
            qa, va = _joint_adr(self.model, name)
            self._qpos_adr[name] = qa
            self._qvel_adr[name] = va

        self._act_adr = {}
        for name in self._arm_action_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self._act_adr[name] = aid

        self._cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.CUBE_BODY_NAME
        )
        self._cube_free_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.CUBE_FREE_JOINT_NAME
        )
        self._grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.GRASP_SITE_NAME
        )
        self._gripper_joint_adr = self._qpos_adr["gripper"]

        self._wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.WRIST_CAMERA_NAME
        )
        if self._wrist_cam_id < 0:
            raise ValueError(
                f"Wrist camera '{self.WRIST_CAMERA_NAME}' not found in MuJoCo model."
            )

        self._topdown_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.TOPDOWN_CAMERA_NAME
        )

        self._step_count = 0
        self._cube_attached = False
        self._cube_grasp_offset = np.zeros(3, dtype=np.float32)
        self._cube_spawn = self.CUBE_START.copy()
        self._target_pos = self.TARGET.copy()

        mujoco.mj_forward(self.model, self.data)

    def _joint_angles(self):
        out = []
        for name in self._joint_names:
            adr = self._qpos_adr[name]
            out.append(float(self.data.qpos[adr]) if adr is not None else 0.0)
        return np.array(out, dtype=np.float32)

    def _cube_position(self):
        if self._cube_id >= 0:
            return self.data.xpos[self._cube_id].copy()
        return np.zeros(3, dtype=np.float32)

    def _grasp_site_position(self):
        if self._grasp_site_id >= 0:
            return self.data.site_xpos[self._grasp_site_id].copy()
        return np.zeros(3, dtype=np.float32)

    def _sample_xy_offset(self, xy_range):
        return np.random.uniform(low=-xy_range, high=xy_range).astype(np.float32)

    def _set_cube_pose(self, cube_pos):
        if self._cube_free_joint < 0:
            return
        qa = self.model.jnt_qposadr[self._cube_free_joint]
        dof = self.model.jnt_dofadr[self._cube_free_joint]
        self.data.qpos[qa:qa + 3] = cube_pos
        self.data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[dof:dof + 6] = 0.0

    def _sync_attached_cube(self):
        if not self._cube_attached or self._cube_free_joint < 0:
            return
        qa = self.model.jnt_qposadr[self._cube_free_joint]
        dof = self.model.jnt_dofadr[self._cube_free_joint]
        grasp_pos = self._grasp_site_position()
        self.data.qpos[qa:qa + 3] = grasp_pos + self._cube_grasp_offset
        self.data.qvel[dof:dof + 6] = 0.0

    def _update_grasp_state(self):
        if self._gripper_joint_adr is None:
            return

        gripper_val = float(self.data.qpos[self._gripper_joint_adr])
        cube_pos = self._cube_position()
        grasp_pos = self._grasp_site_position()
        distance = float(np.linalg.norm(cube_pos - grasp_pos))

        if (
            not self._cube_attached
            and gripper_val >= self.GRASP_CLOSE_THRESHOLD
            and distance <= self.GRASP_DISTANCE_THRESHOLD
        ):
            self._cube_attached = True
            self._cube_grasp_offset[:] = 0.0

        if self._cube_attached and gripper_val <= self.RELEASE_OPEN_THRESHOLD:
            self._cube_attached = False

    def _get_obs(self):
        return np.concatenate(
            [self._joint_angles(), self._cube_position(), self._target_pos],
            dtype=np.float32,
        )

    def reset(
        self,
        randomize_cube=False,
        randomize_target=False,
        cube_pos=None,
        target_pos=None,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        for name, val in HOME_JOINTS.items():
            qa = self._qpos_adr[name]
            if qa is not None:
                self.data.qpos[qa] = val
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        cube_spawn = np.asarray(cube_pos, dtype=np.float32).copy() if cube_pos is not None else self.CUBE_START.copy()
        if randomize_cube and cube_pos is None:
            cube_spawn[:2] += self._sample_xy_offset(self.CUBE_RANDOM_XY_RANGE)
        cube_spawn[2] = self.DEFAULT_CUBE_Z

        target = np.asarray(target_pos, dtype=np.float32).copy() if target_pos is not None else self.TARGET.copy()
        if randomize_target and target_pos is None:
            target[:2] += self._sample_xy_offset(self.TARGET_RANDOM_XY_RANGE)
        target[2] = self.DEFAULT_CUBE_Z

        self._target_pos = target
        self._cube_spawn = cube_spawn
        self._cube_attached = False
        self._cube_grasp_offset[:] = 0.0
        self._step_count = 0

        self._set_cube_pose(cube_spawn)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(f"Expected {self.ACTION_DIM}-dim action, got shape {action.shape}")

        desired = {}
        for i, name in enumerate(self._arm_action_names):
            qa = self._qpos_adr[name]
            if qa is None:
                continue
            target_pos = float(action[i])
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                lo, hi = float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])
                target_pos = float(np.clip(target_pos, lo, hi))
            desired[name] = target_pos

        for _ in range(5):
            for name, pos in desired.items():
                if name in self._act_adr:
                    self.data.ctrl[self._act_adr[name]] = pos
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_state()
            self._sync_attached_cube()
            mujoco.mj_forward(self.model, self.data)

        self._step_count += 1

        cube = self._cube_position()
        dist_xy = float(np.linalg.norm(cube[:2] - self._target_pos[:2]))
        reward = -dist_xy
        done = self._step_count >= self.EPISODE_LEN

        info = {
            "cube_pos": cube.copy(),
            "target": self._target_pos.copy(),
            "target_pos": self._target_pos.copy(),
            "cube_spawn": self._cube_spawn.copy(),
            "distance": dist_xy,
            "step_count": self._step_count,
            "cube_attached": self._cube_attached,
        }
        return self._get_obs(), reward, done, info

    def _render_camera(self, camera_name, width=None, height=None, depth=False):
        width = width or self.RENDER_WIDTH
        height = height or self.RENDER_HEIGHT
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        if depth:
            renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=camera_name)
        image = renderer.render().copy()
        renderer.close()
        return image

    def render_topdown(self, width=None, height=None):
        camera_name = self.TOPDOWN_CAMERA_NAME if self._topdown_cam_id >= 0 else self.WRIST_CAMERA_NAME
        return self._render_camera(camera_name, width=width, height=height, depth=False)

    def render_wrist(self, width=None, height=None):
        return self._render_camera(self.WRIST_CAMERA_NAME, width=width, height=height, depth=False)

    def render_wrist_depth(self, width=None, height=None):
        return self._render_camera(self.WRIST_CAMERA_NAME, width=width, height=height, depth=True)

    def render_depth(self, width=None, height=None):
        return self.render_wrist_depth(width=width, height=height)

    def render(self, mode="rgb_array", width=None, height=None):
        if mode in {"rgb_array", "topdown"}:
            return self.render_topdown(width=width, height=height)
        if mode in {"wrist", "wrist_rgb"}:
            return self.render_wrist(width=width, height=height)
        if mode in {"depth", "wrist_depth"}:
            return self.render_wrist_depth(width=width, height=height)
        raise ValueError(f"Unsupported render mode: {mode}")
