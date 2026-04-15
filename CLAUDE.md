# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before making changes, Claude must check [UPDATE.md](/home/darshan/robothon/physical-ai-challange-2026/UPDATE.md) every time and keep re-checking it during longer work loops so it stays aligned with the latest verified progress.

## Repository Overview

This is the SO-101 Physical AI Hackathon challenge — pick-and-place task. Two major components:

1. **`task1_pick_place/`** — Training and simulation pipeline (pure Python, MuJoCo, PyTorch)
2. **`workshop/dev/docker/`** — Full Docker dev environment with ROS 2 Humble, MoveIt, and Gazebo

## Task 1: Pick-and-Place Pipeline

### Files

- **`environment.py`** — MuJoCo-based `PickPlaceEnv`. Loads `scene.xml` from the `so101_mujoco` ROS package. Observation: 7 joint angles + 3 cube pos + 3 target pos = 13-dim. Action: 6 joint velocities.
- **`collect_demos.py`** — Scripted waypoint controller (10 waypoints, linear interpolation, P-control). Runs 50 episodes, saves `(obs, action)` pairs to `demos.pkl`. Waypoints defined as 6-dof joint arrays `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`. Gripping strategy: close gripper BEFORE touching cube, friction holds it.
- **`train.py`** — Behavior cloning (simple pipeline). `PolicyMLP`: Linear(13→256→256→128→6). SmoothL1 loss, AdamW, CosineAnnealingWarmRestarts. Saves best model by loss to `policy.pt`.
- **`evaluate.py`** — Runs 20 rollouts, measures XY error in mm. Pass threshold: 30mm.

### Two Training Pipelines

**Pipeline A — Scripted (simpler, fully automated):**
```bash
cd /home/hacker/workspace/task1_pick_place
python collect_demos.py          # → demos.pkl (50 episodes)
python train.py                # → policy.pt (200 epochs max, early stopping)
python evaluate.py             # 20 trials, reports pass rate + mean error
```

**Pipeline B — Teleop + Stack (more capable, requires human input):**
```bash
cd /home/hacker/workspace/task1_pick_place
python teleop_data_collector.py   # keyboard teleop → teleop_data.hdf5
python render_teleop_data.py     # optional: render camera frames → teleop_images.hdf5
python train_policy.py           # → policy.pt (200 epochs max, train/val split)
```
Pipeline B uses 3-frame stacked observations (39-dim input), 7-dim actions (6 joints + gripper), weighted joint+gripper loss, and train/val split. Falls back to `demos.pkl` if `teleop_data.hdf5` not found.

### Additional Utilities

- **`mujoco_env_wrapper.py`** — Gymnasium wrapper: frame stacking (3 frames), domain randomization (cube spawn), guardrail (base-to-cube > 0.45m → termination), 7-dim action space (arm + sigmoid gripper). `PerceptionGymWrapper` adds vision-based detection replacing GT cube position.
- **`perception_module.py`** — HSV segmentation → 3D projection via runtime MuJoCo camera pose extraction. EMA smoothing, distance-aware Gaussian noise, confidence scoring, 7% simulated dropout.
- **`demo_perception.py`** — Interactive judge demo. Keyboard teleop OR auto scripted mode with state machine. Shows GT vs detected cube in real-time. Press `A` to toggle auto mode.
- **`debug_gripper.py`** — Visual gripper geometry debugger. Interactive waypoint testing, grid search for approach poses.
- **`render_teleop_data.py`** — Post-process teleop HDF5: render wrist camera images at each step, save aligned (image, obs, action) triplets.

### Workflow

```bash
# Run inside container (see Docker section)
cd /home/hacker/workspace/task1_pick_place

# 1. Collect demonstrations
python collect_demos.py          # → demos.pkl (50 episodes)

# 2. Train policy
python train.py                  # → policy.pt (100 epochs)

# 3. Evaluate
python evaluate.py               # 20 trials, reports pass rate + mean error
```

**Dependencies:** `mujoco`, `torch`, `numpy`, `pickle`, `h5py`, `gymnasium` — available in the Docker container's Python 3.12 venv.

## Docker Environment

**Image:** `vishlandrobotics/physical-ai-challange-2026:latest` (built from `workshop/dev/docker/Dockerfile`)

**Key stack:** NVIDIA CUDA 12.1 + Ubuntu 22.04 + ROS 2 Humble + MoveIt 2 + Gazebo Harmonic + MuJoCo + PyTorch (CUDA) + LeRobot + Python 3.12 (via `uv`)

**ROS packages** (cloned into `workspace/src/`):
- `so101_description` — URDF + meshes for the SO-101 robot
- `so101_mujoco` — MuJoCo simulation (`scene.xml` lives here)
- `so101_moveit_config` — MoveIt2 config for the follower arm
- `so101_leader_moveit_config` — Leader arm MoveIt2 config
- `so101_leader_description` — Leader URDF + meshes
- `so101_gazebo` — Gazebo bridge
- `so101_unified_bringup` — Unified launch with ROS services: `/move_to_joint_states`, `/pick_front`, `/place_object`, etc.

**Important build note:** Always `deactivate` the Python 3.12 venv before running `colcon build`. ROS 2 Humble requires system Python 3.10. Build command:

```bash
deactivate 2>/dev/null; unset VIRTUAL_ENV
source /opt/ros/humble/setup.bash
cd /home/hacker/workspace
colcon build --symlink-install
source install/setup.bash
```

**MuJoCo IK fix:** The default IK solver timeout (0.05s) is too short for the 5-DOF arm. Set `kinematics_solver_timeout: 0.5` in `workspace/src/so101_moveit_config/config/kinematics.yaml` and rebuild before Cartesian moves will work.

**LeRobot integration:** LeRobot is installed at `/opt/lerobot`. The `lerobot_venv` Python 3.12 environment is auto-activated on container login. Re-activate with `source /opt/lerobot_venv/bin/activate`.

## Architecture Notes

- The `PickPlaceEnv` in `task1_pick_place/` is a **headless** simulation — no ROS dependencies. It directly loads `scene.xml` from the `so101_mujoco` package directory.
- `collect_demos.py` uses a **scripted controller** (no ML) to generate training data. The 10-waypoint trajectory with grip-close-BEFORE-touch strategy is tuned for friction-based grasping.
- Policy is purely **imitation learning** (behavior cloning). No RL, no reward shaping.
- Evaluation uses a **fixed target position** (`[0.30, 0.10, 0.0]`), not randomization.
