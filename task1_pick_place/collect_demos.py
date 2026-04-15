#!/usr/bin/env python3
"""
collect_demos.py — Collect pick-and-place demonstrations using scripted controller.
Each demo = (observation, action) pairs for one full pick-place episode.
Saves demos.pkl on completion.

Action space: 6-DOF target joint positions
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
The env's internal P-control tracks these position targets.
"""

import os
import pickle
import numpy as np

try:
    from .environment import PickPlaceEnv
except ImportError:
    from environment import PickPlaceEnv


# ── Waypoint definitions ──────────────────────────────────────────────────────
# Joint angles: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
# GRIPPING: close gripper at waypoint 3 BEFORE touching cube — friction holds it.
# Each waypoint is a DIRECT target position for the policy to output.
# The arm reaches it via the env's internal position control.

WAYPOINTS = [
    # [sp,      sl,      ef,      wf,     wr,   grip]
    [  0.0,   -0.40,   0.20,   0.10,   0.0,   0.00],  # 0: home safe
    [ -0.02,  -1.44,   1.46,   0.79,   0.0,   0.00],  # 1: above cube
    [  0.01,  -0.99,   1.34,   0.80,   0.0,   0.00],  # 2: descend to grasp
    [  0.01,  -0.99,   1.34,   0.80,   0.0,   1.20],  # 3: close gripper
    [ -0.02,  -1.66,   1.32,   0.74,   0.0,   1.20],  # 4: lift cube
    [ -0.38,  -0.18,   0.45,   0.52,   0.0,   1.20],  # 5: move above target
    [ -0.39,  -0.11,   0.75,   0.07,   0.0,   1.20],  # 6: lower to target
    [ -0.39,  -0.11,   0.75,   0.07,   0.0,   0.00],  # 7: release cube
    [ -0.38,  -0.18,   0.45,   0.52,   0.0,   0.00],  # 8: retreat
    [  0.0,   -0.40,   0.20,   0.10,   0.0,   0.00],  # 9: return home
]
WAYPOINTS = np.array(WAYPOINTS, dtype=np.float32)


def interpolate(start, end, t):
    """Linear interpolation between two waypoints."""
    return start + (end - start) * np.clip(t, 0.0, 1.0)


def run_episode(env, num_steps=400, interpolate_steps=35, noise_std=0.0):
    """
    Run one scripted pick-place episode.
    Returns list of (obs, action) tuples.

    noise_std: std dev of Gaussian noise added to target joint positions.
    Positive noise makes the policy robust to small errors.
    """
    obs = env.reset()
    all_data = []
    final_info = None

    # 10 waypoints, each phase = interpolate_steps steps
    phase_len = interpolate_steps
    phases    = len(WAYPOINTS)

    for step in range(num_steps):
        phase = min(step // phase_len, phases - 1)
        t     = (step % phase_len) / phase_len
        target_joints = interpolate(WAYPOINTS[phase], WAYPOINTS[(phase + 1) % phases], t)

        # Add Gaussian noise to target for robustness (only affects stored actions)
        if noise_std > 0:
            noise = np.random.randn(6).astype(np.float32) * noise_std
            # Clip noise so we don't go out of joint limits
            target_joints = np.clip(target_joints + noise, -1.5, 1.5)

        # Action = direct target joint position (position-delta space)
        action = target_joints.astype(np.float32)
        current_obs = obs.copy()
        obs, reward, done, info = env.step(action)
        final_info = info
        all_data.append((current_obs, action.copy()))

        if done:
            break

    return all_data, final_info


def collect_demos(n_episodes=50, output_path="demos.pkl", noise_std=0.0):
    """Collect n demonstration episodes and save to disk."""
    env = PickPlaceEnv()

    all_demos = []
    for ep in range(1, n_episodes + 1):
        episode_data, final_info = run_episode(env, noise_std=noise_std)
        all_demos.extend(episode_data)
        cube_pos = final_info["cube_pos"]
        target = final_info["target"]
        error_mm = np.linalg.norm(cube_pos[:2] - target[:2]) * 1000

        print(f"Episode {ep:3d}/50 complete | Final error: {error_mm:.1f}mm")

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(all_demos, f)

    print(f"\nSaved {len(all_demos)} (obs, action) pairs → {output_path}")
    return all_demos


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos.pkl")
    demos = collect_demos(n_episodes=50, output_path=OUT, noise_std=0.0)
    print(f"collect_demos.py — DONE ({len(demos)} transitions)")
