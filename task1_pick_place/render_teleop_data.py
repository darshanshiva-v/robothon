#!/usr/bin/env python3
"""
render_teleop_data.py — Replay teleop HDF5 and render wrist RGB/depth.
"""

import os
import sys
import json
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PickPlaceEnv


def render_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input HDF5 not found: {input_path}")

    with h5py.File(input_path, "r") as fin:
        all_obs = fin["episodes/obs"][:]
        all_act = fin["episodes/act"][:]
        ep_idx = fin["episodes/episode_idx"][:]

    env = PickPlaceEnv()
    rgb0 = env.render_wrist()
    h, w = rgb0.shape[:2]

    rgb_frames = []
    depth_frames = []
    obs_out = []
    act_out = []
    out_ep_idx = [0]

    for ep in range(len(ep_idx) - 1):
        start, end = int(ep_idx[ep]), int(ep_idx[ep + 1])
        ep_obs = all_obs[start:end]
        ep_act = all_act[start:end]
        env.reset()

        for obs, act in zip(ep_obs, ep_act):
            rgb_frames.append(env.render_wrist())
            depth_frames.append(env.render_wrist_depth())
            obs_out.append(obs.astype(np.float32))
            act_out.append(act.astype(np.float32))
            env_action = act[:6].copy()
            env_action[5] = 1.2 if act[6] > 0.5 else 0.0
            env.step(env_action)

        out_ep_idx.append(len(rgb_frames))

    with h5py.File(output_path, "w") as fout:
        episodes = fout.create_group("episodes")
        episodes.create_dataset("rgb_frames", data=np.asarray(rgb_frames, dtype=np.uint8), dtype="uint8")
        episodes.create_dataset("depth_frames", data=np.asarray(depth_frames, dtype=np.float32), dtype="float32")
        episodes.create_dataset("raw_obs", data=np.asarray(obs_out, dtype=np.float32), dtype="float32")
        episodes.create_dataset("actions", data=np.asarray(act_out, dtype=np.float32), dtype="float32")
        episodes.create_dataset("episode_idx", data=np.asarray(out_ep_idx, dtype=np.int32), dtype="int32")

        metadata = fout.create_group("metadata")
        metadata.attrs["camera_name"] = PickPlaceEnv.WRIST_CAMERA_NAME
        metadata.attrs["image_size"] = [w, h]
        metadata.attrs["episodes_from"] = os.path.basename(input_path)
        metadata.attrs["n_episodes"] = len(ep_idx) - 1
        metadata.attrs["n_total_steps"] = len(rgb_frames)

    summary_path = output_path.replace(".hdf5", "_accuracy.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_episodes": int(len(ep_idx) - 1),
                "n_total_steps": int(len(rgb_frames)),
                "image_size": [int(w), int(h)],
                "camera_name": PickPlaceEnv.WRIST_CAMERA_NAME,
            },
            f,
            indent=2,
        )
    return output_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "teleop_data.hdf5")
    output_path = os.path.join(script_dir, "teleop_images.hdf5")
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    render_dataset(input_path, output_path)
