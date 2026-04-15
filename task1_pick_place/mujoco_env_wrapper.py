#!/usr/bin/env python3
"""
mujoco_env_wrapper.py — Gym wrappers for Task 1.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PickPlaceEnv

try:
    from perception_module import ObjectDetector
    _HAS_PERCEPTION = True
except ImportError:
    ObjectDetector = None
    _HAS_PERCEPTION = False


N_FRAMES = 3
RAW_OBS_DIM = PickPlaceEnv.OBS_DIM
STACKED_OBS_DIM = RAW_OBS_DIM * N_FRAMES
PERCEPTION_FRAME_DIM = 14
PERCEPTION_STACKED_DIM = PERCEPTION_FRAME_DIM * N_FRAMES
ACTION_DIM = 7
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.2
GRIPPER_THRESHOLD = 0.5
GUARDRAIL_THRESHOLD_M = 0.45
GUARDRAIL_PENALTY = -10.0
TARGET_POS = np.array([0.25, 0.20, 0.025], dtype=np.float32)
CUBE_X_MIN, CUBE_X_MAX = 0.15, 0.30
CUBE_Y_MIN, CUBE_Y_MAX = -0.20, 0.20
CUBE_Z = 0.025


def random_cube_pos():
    return np.array(
        [
            np.random.uniform(CUBE_X_MIN, CUBE_X_MAX),
            np.random.uniform(CUBE_Y_MIN, CUBE_Y_MAX),
            CUBE_Z,
        ],
        dtype=np.float32,
    )


class FrameRingBuffer:
    def __init__(self, n_frames, obs_dim):
        self.n_frames = n_frames
        self.obs_dim = obs_dim
        self._buffer = np.zeros((n_frames, obs_dim), dtype=np.float32)
        self._count = 0
        self._pos = 0

    def reset(self, obs):
        self._buffer[:] = np.asarray(obs, dtype=np.float32)
        self._count = self.n_frames
        self._pos = 0

    def push(self, obs):
        self._buffer[self._pos] = np.asarray(obs, dtype=np.float32)
        self._pos = (self._pos + 1) % self.n_frames
        self._count = min(self._count + 1, self.n_frames)

    def get_stacked(self):
        if self._count < self.n_frames:
            return np.tile(self._buffer[0], self.n_frames)
        ordered = np.concatenate((self._buffer[self._pos:], self._buffer[:self._pos]), axis=0)
        return ordered.reshape(-1).astype(np.float32)


class MujocoGymWrapper(gym.Wrapper):
    metadata = {"render_modes": ["rgb_array", "wrist", "depth"]}

    def __init__(self, env=None):
        super().__init__(env or PickPlaceEnv())
        self._frame_buf = FrameRingBuffer(N_FRAMES, RAW_OBS_DIM)
        self._step_count = 0
        self._cube_spawn_pos = None
        self._target_pos = TARGET_POS.copy()
        self._guardrail_triggered = False
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STACKED_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def _reset_raw(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._cube_spawn_pos = random_cube_pos()
        raw_obs = self.env.reset(cube_pos=self._cube_spawn_pos, target_pos=self._target_pos)
        self._step_count = 0
        self._guardrail_triggered = False
        return raw_obs

    def reset(self, seed=None, options=None):
        raw_obs = self._reset_raw(seed=seed)
        self._frame_buf.reset(raw_obs)
        return self._frame_buf.get_stacked(), {
            "guardrail_triggered": False,
            "cube_spawn": self._cube_spawn_pos.copy(),
            "target_pos": self._target_pos.copy(),
        }

    def _map_action(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {action.shape}")
        env_action = action[: PickPlaceEnv.ACTION_DIM].copy()
        env_action[5] = GRIPPER_CLOSED if action[6] > GRIPPER_THRESHOLD else GRIPPER_OPEN
        return env_action

    def step(self, action):
        env_action = self._map_action(action)
        raw_obs, reward, done, info = self.env.step(env_action)
        self._frame_buf.push(raw_obs)
        self._step_count += 1

        cube_pos = raw_obs[7:10]
        dist = float(np.linalg.norm(cube_pos))
        terminated = done
        if dist > GUARDRAIL_THRESHOLD_M:
            reward = GUARDRAIL_PENALTY
            terminated = True
            self._guardrail_triggered = True
            info["guardrail_triggered"] = True
        else:
            info["guardrail_triggered"] = False

        truncated = self._step_count >= PickPlaceEnv.EPISODE_LEN
        info["raw_obs"] = raw_obs.copy()
        info["cube_pos"] = cube_pos.copy()
        info["target_pos"] = self._target_pos.copy()
        info["base_to_cube_dist"] = dist
        return self._frame_buf.get_stacked(), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)


class PerceptionGymWrapper(MujocoGymWrapper):
    def __init__(self, env=None, use_perception=True, debug_overlay=False):
        super().__init__(env=env)
        self._use_perception = use_perception
        self._debug_overlay = debug_overlay
        self._detector = ObjectDetector() if (use_perception and _HAS_PERCEPTION) else None
        self._per_frame_buf = FrameRingBuffer(N_FRAMES, PERCEPTION_FRAME_DIM)
        self._detected_cube_pos = None
        self._detection_confidence = 0.0
        self._last_debug_img = None
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(PERCEPTION_STACKED_DIM,),
            dtype=np.float32,
        )

    def _build_perception_frame(self, raw_obs, cube_pos, confidence):
        safe_cube = np.asarray(cube_pos if cube_pos is not None else raw_obs[7:10], dtype=np.float32)
        safe_cube = np.where(np.isfinite(safe_cube), safe_cube, raw_obs[7:10]).astype(np.float32)
        conf = np.array([float(np.clip(confidence, 0.0, 1.0))], dtype=np.float32)
        return np.concatenate([raw_obs[:7], safe_cube, self._target_pos, conf]).astype(np.float32)

    def _detect(self):
        if self._detector is None:
            return {
                "cube_pos": self.env._cube_position(),
                "confidence": 1.0,
                "debug_overlay": None,
            }
        rgb = self.env.render_wrist()
        depth = self.env.render_wrist_depth()
        return self._detector.detect_from_camera(rgb, depth_buffer=depth, env=self.env, use_depth=True)

    def reset(self, seed=None, options=None):
        raw_obs = self._reset_raw(seed=seed)
        if self._detector is not None:
            self._detector.reset()
        det = self._detect()
        self._detected_cube_pos = np.asarray(det["cube_pos"], dtype=np.float32)
        self._detection_confidence = float(det["confidence"])
        self._last_debug_img = det.get("debug_overlay")
        per_frame = self._build_perception_frame(raw_obs, self._detected_cube_pos, self._detection_confidence)
        self._per_frame_buf.reset(per_frame)
        return self._per_frame_buf.get_stacked(), {
            "guardrail_triggered": False,
            "cube_spawn": self._cube_spawn_pos.copy(),
            "target_pos": self._target_pos.copy(),
            "confidence": self._detection_confidence,
        }

    def step(self, action):
        env_action = self._map_action(action)
        raw_obs, reward, done, info = self.env.step(env_action)
        self._step_count += 1

        det = self._detect()
        self._detected_cube_pos = np.asarray(det["cube_pos"], dtype=np.float32)
        self._detection_confidence = float(det["confidence"])
        self._last_debug_img = det.get("debug_overlay")

        per_frame = self._build_perception_frame(raw_obs, self._detected_cube_pos, self._detection_confidence)
        self._per_frame_buf.push(per_frame)

        dist = float(np.linalg.norm(self._detected_cube_pos))
        terminated = done
        if dist > GUARDRAIL_THRESHOLD_M:
            reward = GUARDRAIL_PENALTY
            terminated = True
            self._guardrail_triggered = True
            info["guardrail_triggered"] = True
        else:
            info["guardrail_triggered"] = False

        truncated = self._step_count >= PickPlaceEnv.EPISODE_LEN
        info["cube_detected"] = self._detected_cube_pos.copy()
        info["cube_gt"] = raw_obs[7:10].copy()
        info["confidence"] = self._detection_confidence
        info["raw_obs"] = raw_obs.copy()
        info["base_to_cube_dist"] = dist
        if self._debug_overlay and self._last_debug_img is not None:
            info["debug_overlay"] = self._last_debug_img.copy()
        return self._per_frame_buf.get_stacked(), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode == "debug" and self._debug_overlay and self._last_debug_img is not None:
            return self._last_debug_img
        return super().render(mode=mode)


def register_gym_env():
    gym.register(id="MujocoGymWrapper-v0", entry_point="mujoco_env_wrapper:MujocoGymWrapper")
    gym.register(id="PerceptionGymWrapper-v0", entry_point="mujoco_env_wrapper:PerceptionGymWrapper")


if __name__ == "__main__":
    env = MujocoGymWrapper(PickPlaceEnv())
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Reset obs shape: {obs.shape}")
