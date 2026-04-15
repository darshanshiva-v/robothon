# Pick-and-Place ML Pipeline — Design Spec
**Date:** 2026-04-14
**Status:** Approved

---

## Overview

Three deliverables build on existing `task1_pick_place/PickPlaceEnv`. The system uses a pure state-based policy (no vision) with position control for the arm and binary gripper.

---

## Shared Constants

| Constant | Value |
|---|---|
| Raw obs dim | 13 (7 joints + 3 cube pos + 3 target pos) |
| Stacked obs dim | 39 (3 frames × 13) |
| Action dim | 7 (6 arm joints + 1 binary gripper) |
| Arm joint count | 6 |
| Data collection rate | 30 Hz |
| Cube spawn envelope | X∈[0.15, 0.30], Y∈[-0.20, 0.20], Z=0.025 |
| Target drop zone | [0.25, 0.20, 0.025] (static, independent of cube) |
| Guardrail threshold | 0.45 m (Euclidean base-to-cube) |
| Guardrail penalty | -10.0 |
| Gripper mapping | binary {0,1} → continuous {0.0, 1.2} |
| Gripper threshold | 0.5 (sigmoid output) |
| Frame stack size | 3 |

---

## Deliverable 1 — teleop_data_collector.py

### Purpose
Allow user to drive the SO-101 arm via keyboard and record (obs, action) pairs during successful pick-and-place episodes.

### Controls

| Key | Action |
|---|---|
| W / S | shoulder_pan ±step |
| E / Q | shoulder_lift ±step |
| R / F | elbow_flex ±step |
| T / G | wrist_flex ±step |
| Y / H | wrist_roll ±step |
| G | toggle gripper (0 ↔ 1.2) |
| Enter | toggle is_recording |
| Space | discard current episode buffer |
| Escape | quit + save all |

### Data Flow

1. `threading.Timer(1/30, callback)` fires every 33ms.
2. If `is_recording == True`: read current joint target (7-dim) + call `env._get_obs()` (13-dim) → append to episode buffer.
3. User presses Enter → `is_recording = False` → episode saved to HDF5.
4. Episodes accumulate across session. On Escape, flush all pending.

### HDF5 Structure

```
/episodes/
  obs       (N_total, 13) float32
  act       (N_total, 7)  float32
  episode_idx  (N_episodes+1,) int32  — cumulative start indices
/metadata/
  obs_dim   int
  act_dim   int
  rate_hz   int
```

### Key Implementation Notes

- Joint targets stored as 7-dim: `[j0, j1, j2, j3, j4, j5, gripper_binary]`.
- `env._get_obs()` directly reads raw 13-dim state (no wrapper needed).
- `threading.Timer` avoids blocking the main loop.
- MuJoCo viewer renders in real-time so user can see the arm while driving.

---

## Deliverable 2 — mujoco_env_wrapper.py

### Class

`MujocoGymWrapper(gymnasium.Wrapper)` wrapping `PickPlaceEnv`.

### reset()

1. Spawn cube at random `(x, y)` in envelope, z=0.025.
2. Target fixed at `[0.25, 0.20, 0.025]` (independent of cube — no coupling).
3. Call `env.reset()` which resets arm to home.
4. Initialize ring buffer with 3 copies of first obs (39-dim).
5. Return stacked obs (39-dim).

### step(action: np.ndarray shape (7,))

1. Extract gripper_val = `1.2 if action[6] > 0.5 else 0.0`.
2. Build full 7-dim action: `arm_joints (action[0:6]) + [gripper_val]`.
3. Call `env.step(full_action)`.
4. Update ring buffer: push new obs, pop oldest.
5. Guardrail check: `dist = np.linalg.norm(cube_pos - [0,0,0])`. If `dist > 0.45`: return penalty + terminated.
6. Return stacked obs (39-dim), reward, terminated, info.

### Observation Space

`gymnasium.spaces.Box(low=-inf, high=inf, shape=(39,), dtype=np.float32)`

### Action Space

`gymnasium.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)` — arm joints normalized, gripper sigmoid expectation.

---

## Deliverable 3 — train_policy.py

### Network Architecture

```
Input (39,)
  → Linear(39, 256) → LayerNorm(256) → ReLU → Dropout(0.05)
  → Linear(256, 256) → LayerNorm(256) → ReLU → Dropout(0.05)
  → Linear(256, 128) → ReLU
  → Linear(128, 7)
  Output: [arm_joints (6), gripper_sigmoid (1)]
```

### Loss Function

```
joint_loss  = SmoothL1Loss(output[:, 0:6], target[:, 0:6])
gripper_loss = BCEWithLogitsLoss(sigmoid(output[:, 6]), target[:, 6])
total_loss  = 0.9 * joint_loss + 0.1 * gripper_loss
```

### Training Loop

- Optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-5)
- Batch size: 64
- Epochs: 200, early stopping (patience=30 on val loss)
- Gradient clipping: max_norm=1.0
- Train/val split: 80/20
- Obs normalization: compute mean/std from training data, apply at forward pass
- Best checkpoint saved to `policy.pt`

### Data Loading

Read HDF5 from Deliverable 1. Numpy arrays → torch tensors. Shuffle train set each epoch.

---

## File Locations

| File | Path |
|---|---|
| Teleop collector | `task1_pick_place/teleop_data_collector.py` |
| Gym wrapper | `task1_pick_place/mujoco_env_wrapper.py` |
| Training script | `task1_pick_place/train_policy.py` |
| Demo data | `task1_pick_place/teleop_data.hdf5` |

All three files live in `task1_pick_place/` alongside existing `environment.py`, `collect_demos.py`, `train.py`.