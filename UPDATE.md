# UPDATE.md

This file tracks the latest Task 1 integration work and what was actually verified.

## What Changed

### Shared environment contract

- Rebuilt `task1_pick_place/environment.py` as the canonical Task 1 API.
- Locked the base observation to `13` dims and the base environment action to `6` dims.
- Moved reset ownership into `PickPlaceEnv.reset(...)` so cube and target placement are configured in one place.
- Added explicit wrist RGB, wrist depth, and top-down render helpers used by wrappers and demos.
- Kept the working grasp / release behavior and the longer `400`-step episode horizon.

### Scripted pipeline

- Kept `collect_demos.py` aligned with the new environment contract and fixed its final-error reporting to use the true final environment state.
- Updated `train.py` to save a structured checkpoint payload plus `policy_scripted_config.json`.
- Updated `evaluate.py` to load either the new structured checkpoint or a legacy raw state dict.
- Added working `--random` and `--trials` CLI options to evaluation.

### Teleop pipeline

- Replaced the old teleop collector with a deterministic keyboard collector that has no gripper keybinding conflict.
- Locked the HDF5 schema to:
  - `/episodes/obs`
  - `/episodes/act`
  - `/episodes/episode_idx`
- Split teleop training outputs from the scripted baseline:
  - `policy_teleop.pt`
  - `policy_teleop_config.json`
- Removed silent fallback from `train_policy.py`; it now fails clearly if `teleop_data.hdf5` is missing.

### Wrapper / perception / utilities

- Rebuilt `mujoco_env_wrapper.py` around the new environment contract.
- Standardized wrapper action mapping to `7` dims with an explicit binary gripper channel.
- Standardized stacked observation sizes:
  - `39` dims for raw-state wrapper
  - `42` dims for perception wrapper
- Made `perception_module.py` degrade safely when OpenCV is unavailable instead of crashing mid-run.
- Reworked `render_teleop_data.py` to use the environment’s wrist render methods directly.
- Added `task1_pick_place/__init__.py` so the directory can be imported during smoke tests.
- Added `task1_pick_place/README.md` as the canonical Task 1 runbook.

## What Was Verified

### Static verification

- `python3 -m py_compile` passed for the updated Task 1 Python files.

### Runtime verification completed in this shell

- `PickPlaceEnv.reset(...)`, `step(...)`, `render_wrist()`, and `render_wrist_depth()` all executed successfully.
- A one-episode scripted collection run completed successfully and finished at about `5.4 mm` final XY error.
- The perception module returned finite fallback detections and confidence values without crashing when OpenCV was unavailable.
- A lightweight scripted train/evaluate smoke test ran end-to-end on temporary data:
  - training loop completed
  - policy evaluation executed without interface errors
  - the tiny smoke run did **not** converge to a passing placement result

## Verification Gaps

- `gymnasium` was not installed in the active host shell, so wrapper runtime verification stopped at dependency import.
- `h5py` was not installed in the active host shell, so teleop HDF5 render-path verification could not be completed there.
- Full scripted baseline verification with a fresh full dataset and a full training run is still recommended before calling the benchmark path fully validated.

## Recommended Next Commands

Run these in an environment with the required Python packages installed:

```bash
python3 task1_pick_place/collect_demos.py
python3 task1_pick_place/train.py
python3 task1_pick_place/evaluate.py
python3 task1_pick_place/evaluate.py --random
python3 task1_pick_place/teleop_data_collector.py
python3 task1_pick_place/train_policy.py
python3 task1_pick_place/render_teleop_data.py
python3 task1_pick_place/mujoco_env_wrapper.py
python3 task1_pick_place/demo_perception.py
```
