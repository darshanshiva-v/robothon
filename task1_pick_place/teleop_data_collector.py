#!/usr/bin/env python3
"""
teleop_data_collector.py — Keyboard teleoperation data collection for Task 1.

Dataset contract:
  /episodes/obs         (N, 13) float32
  /episodes/act         (N, 7)  float32  -> [6 joint targets, gripper_binary]
  /episodes/episode_idx (E+1,) int32
"""

import os
import sys
import time
import numpy as np
import h5py
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PickPlaceEnv
from keyboard_utils import RawKeyboard


RATE_HZ = 30.0
DT = 1.0 / RATE_HZ
STEP_SIZE = 0.05
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.2
ACTION_DIM = 7
OBS_DIM = 13

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teleop_data.hdf5")

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

KEY_TO_JOINT = {
    "LEFT": (0, STEP_SIZE),
    "RIGHT": (0, -STEP_SIZE),
    "UP": (1, STEP_SIZE),
    "DOWN": (1, -STEP_SIZE),
    "a": (2, STEP_SIZE),
    "z": (2, -STEP_SIZE),
    "s": (3, STEP_SIZE),
    "x": (3, -STEP_SIZE),
    "d": (4, STEP_SIZE),
    "c": (4, -STEP_SIZE),
}

KEY_TOGGLE_GRIPPER = " "
KEY_TOGGLE_RECORD = {"ENTER"}
KEY_DISCARD = "BACKSPACE"
KEY_QUIT = {"ESC"}


def append_episode_hdf5(output_path, ep_obs, ep_act):
    with h5py.File(output_path, "a") as f:
        episodes = f.require_group("episodes")
        metadata = f.require_group("metadata")
        metadata.attrs["obs_dim"] = OBS_DIM
        metadata.attrs["act_dim"] = ACTION_DIM
        metadata.attrs["rate_hz"] = RATE_HZ
        metadata.attrs["step_size"] = STEP_SIZE
        metadata.attrs["action_layout"] = "joint_targets_6_plus_gripper_binary"

        n_new = int(ep_obs.shape[0])
        if "obs" not in episodes:
            episodes.create_dataset("obs", data=ep_obs, maxshape=(None, OBS_DIM), dtype="float32")
            episodes.create_dataset("act", data=ep_act, maxshape=(None, ACTION_DIM), dtype="float32")
            episodes.create_dataset("episode_idx", data=np.array([0, n_new], dtype=np.int32), maxshape=(None,), dtype="int32")
            return

        old_n = int(episodes["obs"].shape[0])
        episodes["obs"].resize(old_n + n_new, axis=0)
        episodes["obs"][old_n:] = ep_obs
        episodes["act"].resize(old_n + n_new, axis=0)
        episodes["act"][old_n:] = ep_act

        ep_idx = episodes["episode_idx"][:]
        new_ep_idx = np.append(ep_idx, old_n + n_new).astype(np.int32)
        episodes["episode_idx"].resize(new_ep_idx.shape[0], axis=0)
        episodes["episode_idx"][:] = new_ep_idx


def current_action(joint_targets, gripper_closed):
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:6] = joint_targets
    action[6] = 1.0 if gripper_closed else 0.0
    return action


def main():
    print("=" * 60)
    print("SO-101 Teleop Collector")
    print("Controls:")
    print("  LEFT/RIGHT shoulder_pan   UP/DOWN shoulder_lift")
    print("  A/Z elbow_flex           S/X wrist_flex      D/C wrist_roll")
    print("  SPACE toggle gripper")
    print("  ENTER start/stop recording   BACKSPACE discard current episode   ESC quit")
    print("=" * 60)

    env = PickPlaceEnv()
    env.reset()

    joint_targets = np.zeros(6, dtype=np.float32)
    gripper_closed = False
    is_recording = False
    episode_obs = []
    episode_act = []
    last_record_ts = time.monotonic()
    keyboard = RawKeyboard()

    viewer = mujoco.viewer.launch_passive(
        env.model,
        env.data,
        show_left_ui=False,
        show_right_ui=False,
    )
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -25
    viewer.cam.distance = 1.4
    viewer.cam.lookat = np.array([0.2, 0.0, 0.1], dtype=np.float64)

    try:
        while viewer.is_running():
            key = keyboard.read_key()
            if key is not None:
                if key in KEY_TO_JOINT:
                    idx, delta = KEY_TO_JOINT[key]
                    joint_targets[idx] += delta
                    print(f"[{JOINT_NAMES[idx]}] -> {joint_targets[idx]:.3f}")
                elif key == KEY_TOGGLE_GRIPPER:
                    gripper_closed = not gripper_closed
                    state = "CLOSED" if gripper_closed else "OPEN"
                    print(f"[gripper] -> {state}")
                elif key in KEY_TOGGLE_RECORD:
                    if is_recording:
                        is_recording = False
                        if episode_obs:
                            ep_obs = np.asarray(episode_obs, dtype=np.float32)
                            ep_act = np.asarray(episode_act, dtype=np.float32)
                            append_episode_hdf5(OUTPUT_PATH, ep_obs, ep_act)
                            print(f"[saved] {len(ep_obs)} samples -> {OUTPUT_PATH}")
                        else:
                            print("[stop] no samples recorded")
                    else:
                        episode_obs = []
                        episode_act = []
                        is_recording = True
                        last_record_ts = time.monotonic()
                        print("[recording] started")
                elif key == KEY_DISCARD:
                    episode_obs = []
                    episode_act = []
                    is_recording = False
                    print("[discard] cleared current episode buffer")
                elif key in KEY_QUIT:
                    print("[quit]")
                    break

            env_action = joint_targets.copy()
            env_action[5] = GRIPPER_CLOSED if gripper_closed else GRIPPER_OPEN
            obs, _, _, _ = env.step(env_action)

            now = time.monotonic()
            if is_recording and (now - last_record_ts) >= DT:
                episode_obs.append(obs.copy())
                episode_act.append(current_action(joint_targets, gripper_closed))
                last_record_ts = now

            viewer.sync()
            time.sleep(0.01)
    finally:
        keyboard.close()
        viewer.close()
        if is_recording and episode_obs:
            ep_obs = np.asarray(episode_obs, dtype=np.float32)
            ep_act = np.asarray(episode_act, dtype=np.float32)
            append_episode_hdf5(OUTPUT_PATH, ep_obs, ep_act)
            print(f"[saved on exit] {len(ep_obs)} samples -> {OUTPUT_PATH}")

        if os.path.exists(OUTPUT_PATH):
            with h5py.File(OUTPUT_PATH, "r") as f:
                n_total = int(f["episodes/obs"].shape[0])
                n_eps = int(f["episodes/episode_idx"].shape[0] - 1)
                print(f"Final dataset: {n_total} samples across {n_eps} episodes")


if __name__ == "__main__":
    main()
