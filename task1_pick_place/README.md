# Task 1 Pick-and-Place

This directory now has two supported pipelines built on the shared `PickPlaceEnv` contract in `environment.py`.

## Shared Contract

- Observation: `13` floats = `7` joint values + `3` cube position + `3` target position
- Environment action: `6` floats = `5` arm joints + `1` continuous gripper target
- Teleop / Gym action: `7` floats = environment action layout plus a binary gripper channel for datasets and policies
- Reset owns cube/target placement and supports deterministic or randomized episodes
- Wrist RGB and wrist depth rendering are available from `PickPlaceEnv`

## Scripted Baseline

Run from the repo root:

```bash
python3 task1_pick_place/collect_demos.py
python3 task1_pick_place/train.py
python3 task1_pick_place/evaluate.py
python3 task1_pick_place/evaluate.py --random
```

Outputs:

- `demos.pkl`
- `policy.pt`
- `policy_scripted_config.json`

## Teleop Pipeline

Run from the repo root:

```bash
python3 task1_pick_place/teleop_data_collector.py
python3 task1_pick_place/train_policy.py
python3 task1_pick_place/render_teleop_data.py
```

Outputs:

- `teleop_data.hdf5`
- `policy_teleop.pt`
- `policy_teleop_config.json`
- `teleop_images.hdf5`

## Wrapper / Perception Utilities

```bash
python3 task1_pick_place/mujoco_env_wrapper.py
python3 task1_pick_place/demo_perception.py
python3 task1_pick_place/debug_gripper.py
```

## Dependencies

- Required for the scripted baseline: `mujoco`, `numpy`, `torch`
- Required for the Gym wrappers: `gymnasium`
- Required for teleop dataset rendering: `h5py`
- Required for HSV perception overlays: `opencv-python`

The host shell used during implementation had `mujoco`, `numpy`, and `torch`, but did not have `gymnasium` or `h5py`, so wrapper and teleop-render runtime verification should be done in the Docker environment or another Python environment with those packages installed.
