# Perception Module — Design Spec
**Date:** 2026-04-14
**Status:** Approved

---

## Overview

Integrate vision-based object detection into the MuJoCo pick-place pipeline. Replace ground-truth cube positions with perception-derived positions. Maintain observation space compatibility with existing policy (39-dim stacked). Hackathon timeframe: ≤24h.

---

## Camera Setup

- **Sensor:** wrist-mounted d435i (already in `scene.xml`)
- **MuJoCo camera name:** `d435i`
- **Intrinsics from XML:** `fovy=55°`, assume `640×480` resolution
- **Computed:** `fx = fy = h/2 / tan(fov_y/2)`, `cx = w/2`, `cy = h/2`
- **Perception pipeline:** run every env step (after substeps complete)

---

## perception_module.py

### Class: `ObjectDetector`

```
__init__(self)
  init HSV thresholds hard-coded (red for cube, green for target)
  init camera intrinsics from d435i params
  init EMA state: self._cube_ema = None, self._target_ema = None
  init history buffers: self._cube_history, self._target_history

detect_from_image(rgb, depth=None) → {
    "cube_px": (u, v) or None,
    "target_px": (u, v) or None,
    "cube_detected": bool,
    "target_detected": bool,
    "cube_area": float,
    "target_area": float,
}
```

### HSV Segmentation

- Red range: `[(0, 100, 100), (10, 255, 255)]` + `[(160, 100, 100), (180, 255, 255)]` (two ranges for wrap-around)
- Green range: `[(45, 80, 80), (85, 255, 255)]`
- Morphology: `cv2.erode` (kernel 3×3, 1 iter) then `cv2.dilate` (kernel 5×5, 2 iter)
- Area threshold: min 100 px, max 50000 px
- Find largest contour per color, return centroid

### pixel_to_3d(u, v, depth_or_z, camera_params, use_depth=True)

```
Input:
  u, v         pixel coordinates (float)
  depth_or_z   depth buffer (H×W) if use_depth=True, else single Z value
  camera_params: {fx, fy, cx, cy, width, height}
  use_depth    bool

If use_depth and depth_or_z is array:
  z_cam = linearize_depth(depth_buffer[v, u])   # near=0.05, far=3.0
  x_cam = (u - cx) / fx * z_cam
  y_cam = (v - cy) / fy * z_cam
  # Camera-to-world rotation (wristcam euler from XML)
  R = rotation_matrix(euler)                    # from d435i_link euler
  (x,y,z)_world = R @ (x_cam, y_cam, z_cam) + t_world_cam

Else (planar fallback):
  Ray: origin = cam_pos, direction = (u-cx)/fx, (v-cy)/fy, 1.0 (normalized)
  Intersect with table plane z=TABLE_Z (0.025)
  t = (TABLE_Z - cam_z) / ray_dir_z
  (x,y) = cam_xy + t * ray_dir_xy

Return: (x_world, y_world, z_world) or None if intersection fails
```

**Camera extrinsics (from d435i_link in so101_new_calib.xml):**
- Body pos: `[-0.0225, 0.0558, -0.0141]` (camera mount offset from gripperframe)
- Euler: `[1.2217, 0, 1.5708]` (camera orientation)
- Gripper frame position: read from MuJoCo `gripperframe` site each step
- Full camera world pos = gripperframe_pos + R_gripper @ camera_mount_offset

### Confidence Score

```
confidence = sigmoid((area - min_area) / (max_area - min_area) * 10 - 5)
+ temporal stability bonus if EMA diff < 0.01m
+ detection history bonus (≥3 consecutive frames → +0.1)
Clamp to [0, 1]
```

### EMA Smoothing

```
alpha = 0.3  (current weight)
if ema is None:
    ema = raw_3d_pos
else:
    ema = alpha * raw_3d_pos + (1 - alpha) * ema
```

### Noise Injection

```
sigma_base = 0.005  # 5mm base noise
sigma_dist = sigma_base * (1 + k * dist_from_camera)   # distance-aware
sigma = min(sigma_dist, 0.05)   # cap at 50mm

With prob_dropout=0.07:
    return ema + stale_noise * np.random.randn(3)
    # stale_noise = 0.02 (20mm staleness artifact)

Else:
    return ema + np.random.randn(3) * sigma
```

### Fallback on Detection Failure

```
if not detected:
    if history buffer has last known pos:
        return last_known_pos + small drift noise (sigma/10)
    else:
        return placeholder [nan, nan, nan] and set confidence=0
```

---

## mujoco_env_wrapper.py — Perception Integration

### New class: `PerceptionGymWrapper(MujocoGymWrapper)`

```
__init__(self, env=None, perception_module=None, use_perception=True,
         debug_overlay=False)
  super().__init__(env)
  self.perception = perception_module or ObjectDetector()
  self.use_perception = use_perception
  self.debug_overlay = debug_overlay
  self._perception_debug_img = None

reset()
  # existing randomization
  # then: if use_perception, detect initial cube position
  self._detected_cube_pos = self.perception.detect(env.render_wrist())
  self._ema_cube = self._detected_cube_pos  # init EMA

step(action)
  # existing env.step + frame buffer update
  # then: run perception on latest camera frame
  rgb = self.env.render_wrist()
  det = self.perception.detect_from_image(rgb)
  pos_3d = self.perception.pixel_to_3d(det["cube_px"], ...)
  pos_3d = self.perception.apply_ema(pos_3d, self._ema_cube)
  pos_3d = self.perception.inject_noise(pos_3d)
  self._ema_cube = pos_3d
  confidence = self.perception.compute_confidence(det, self._ema_cube)

  # Replace raw obs [7:10] with detected cube position + noise
  # raw_obs[10:13] stays fixed target (or also detected if target_marker enabled)
  # Append confidence as scalar → per-frame obs = 14-dim
  # Frame stacking → 42-dim (3 × 14)

  info["detected_cube"] = pos_3d
  info["gt_cube"] = raw_obs[7:10].copy()
  info["detection_confidence"] = confidence
  info["debug_rgb"] = self._perception_debug_img  # if debug_overlay
```

### Observation Space (updated)

- **Per-frame:** `[joints(7), cube_pos(3), target_pos(3), confidence(1)]` = 14-dim
- **Stacked (3 frames):** 42-dim
- **Action space:** unchanged (7-dim)

---

## render_teleop_data.py

```
Input:  teleop_data.hdf5
Output: teleop_images.hdf5

For each episode:
  Replay env step-by-step
  rgb = env.render_wrist()
  depth = env.render_wrist_depth()  # attempt
  Store [rgb, depth, step_idx] in aligned HDF5
```

---

## Training Integration

**Preferred: collect demos with perception enabled from day one.**
- Teleop collector logs raw obs (GT) AND detected obs (perception output)
- Policy trained on detected positions
- Both logged for analysis

**Backup: fine-tune existing policy**
- Load existing policy.pt
- Run 20-30 episodes with noisy detected positions
- Fine-tune final layers (few epochs, low LR)

---

## Validation Tests

1. `perception_module.py` standalone: render scene → detect red cube → project to 3D → compare with GT from env
2. End-to-end: run 20 episodes with perception vs without, compare success rates
3. Noise robustness: inject varying sigma, measure XY placement error
4. Detection failure: simulate missed frames, verify fallback works

---

## File Locations

| File | Path |
|---|---|
| Perception module | `task1_pick_place/perception_module.py` |
| Gym wrapper (updated) | `task1_pick_place/mujoco_env_wrapper.py` |
| Render pass | `task1_pick_place/render_teleop_data.py` |
| Config constants | `task1_pick_place/perception_config.py` |

---

## Key Constants

```python
# Camera
IMAGE_W, IMAGE_H = 640, 480
FOVY = 55.0  # degrees from d435i XML
FX = FY = IMAGE_H / (2 * tan(radians(FOVY)/2))  # ~567
CX, CY = IMAGE_W/2, IMAGE_H/2  # 320, 240

# Detection
RED_HUE_LOW  = np.array([[[0, 100, 100], [10, 255, 255]]])
RED_HUE_HIGH = np.array([[[160, 100, 100], [180, 255, 255]]])
GREEN_HUE    = np.array([[[45, 80, 80], [85, 255, 255]]])
AREA_MIN, AREA_MAX = 100, 50000

# 3D projection
TABLE_Z = 0.025
CAMERA_NEAR = 0.05
CAMERA_FAR = 3.0

# EMA
EMA_ALPHA = 0.3

# Noise
SIGMA_BASE = 0.005  # 5mm
SIGMA_K = 0.5        # distance scaling
DROP_PROB = 0.07
STALE_NOISE = 0.02  # 20mm
```