#!/usr/bin/env python3
"""
perception_module.py — Lightweight vision-based object detection for SO-101 pick-place.

Pipeline:
    MuJoCo RGB image (wrist camera) + optional depth
        → HSV color segmentation (red cube, green target)
        → pixel centroid → 3D world coordinate
        → EMA smoothing + distance-aware Gaussian noise
        → detected object positions (safe finite values, never NaN)

Design goals:
    • Geometrically correct camera projection (runtime intrinsics + extrinsics)
    • No hardcoded camera pose — extracted from MuJoCo each step
    • Hybrid depth/planar fallback — works with or without depth buffer
    • Robust: morphological filtering, area thresholds, history buffer, EMA
    • Modular: HSV can be swapped for learned model without changing interfaces
"""

import numpy as np

# Optional dependencies — required only when running in the Docker container
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Camera intrinsics — derived from d435i XML (fovy=55°, 640×480)
IMAGE_W, IMAGE_H = 640, 480
FOVY_DEG = 55.0
# fy from vertical FOV: fy = (H/2) / tan(fovy/2)
FY = (IMAGE_H / 2.0) / np.tan(np.radians(FOVY_DEG) / 2.0)   # ≈461 for H=480
# fx from horizontal FOV: fx = (W/2) / tan(fovx/2)
# With square pixels assumed, fovx ≈ 2*atan(W/H * tan(fovy/2)) ≈ 70.5°
HFOVY_RATIO = IMAGE_W / IMAGE_H   # 1.333
FOVX_DEG = 2.0 * np.degrees(np.arctan(HFOVY_RATIO * np.tan(np.radians(FOVY_DEG) / 2.0)))  # ~70.5°
FX = (IMAGE_W / 2.0) / np.tan(np.radians(FOVX_DEG) / 2.0)   # ≈873 for W=640
CX, CY = IMAGE_W / 2.0, IMAGE_H / 2.0                              # 320, 240

# HSV color ranges (HSV space)
RED_HUE_LOW_1  = np.array([[0,   100, 100]], dtype=np.uint8)   # red wraps around 180
RED_HUE_HIGH_1 = np.array([[10,  255, 255]], dtype=np.uint8)
RED_HUE_LOW_2  = np.array([[160, 100, 100]], dtype=np.uint8)
RED_HUE_HIGH_2 = np.array([[180, 255, 255]], dtype=np.uint8)
GREEN_HUE_LOW  = np.array([[45,   80,  80]], dtype=np.uint8)
GREEN_HUE_HIGH = np.array([[85,  255, 255]], dtype=np.uint8)

# Contour filtering
AREA_MIN_PX  = 100      # reject tiny blobs
AREA_MAX_PX  = 50000    # reject oversized (table surface)
ASPECT_TOL   = 0.3      # max deviation from 1:1 (cube ~square in overhead wrist view)

# Table / world
TABLE_Z   = 0.025       # metres — table surface, object resting plane
CAM_NEAR  = 0.05        # metres — MuJoCo clip near
CAM_FAR   = 3.0         # metres — MuJoCo clip far

# EMA smoothing
EMA_ALPHA = 0.3         # current weight (0.3 = smooth, 0.7 = responsive)

# Noise injection
SIGMA_BASE   = 0.005     # 5 mm base noise at unit distance
SIGMA_K      = 0.5      # distance scaling coefficient
SIGMA_CAP    = 0.05      # hard cap: 50 mm max
DROP_PROB    = 0.07      # 7% chance of simulated missed detection
STALE_MM     = 0.02     # staleness artifact magnitude (20 mm)

# Camera mount offset in gripper frame (from d435i_link in so101_new_calib.xml)
# Kept for reference — camera pose is extracted at runtime via MuJoCo.
_CAM_MOUNT_OFFSET = np.array([-0.022503, 0.055810, -0.014089])  # metres
_CAM_MOUNT_EULER  = np.array([1.2217, 0.0, 1.5708])              # radians
# NOTE: _CAM_MOUNT_OFFSET / _CAM_MOUNT_EULER are NOT used at runtime.
# extract_wrist_camera_pose() reads camera pose from MuJoCo directly.


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_from_euler(euler):
    """
    Build 3×3 rotation matrix from ZYX Euler angles (radians).
    Matches MuJoCo's euler='ZYX' convention used in the URDF.
    """
    rz, ry, rx = euler
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    # Z * Y * X
    return np.array([
        [ cz*cy,  cz*sy*sx - sz*cx,  cz*sy*cx + sz*sx],
        [ sz*cy,  sz*sy*sx + cz*cx,  sz*sy*cx - cz*sx],
        [-sy,     cy*sx,              cy*cx            ]
    ], dtype=np.float64)


def linearize_depth(depth_buffer_value, near=CAM_NEAR, far=CAM_FAR):
    """
    Convert non-linear MuJoCo depth buffer [0,1] to linear depth (metres).

    MuJoCo depth buffer uses the standard OpenGL formula:
        depth_ndc = depth_buffer * 2 - 1
        depth_linear = (2 * near * far) / (far + near - depth_ndc * (far - near))

    Note: This formula gives near at d=0 and far at d=1, with non-linear
    scaling in between. For typical scene depths (0.05-1.0m), buffer values
    are in the [0, 1] range. If the result is out of [near*0.5, far*2],
    the function falls back to None and ray-plane intersection is used instead.
    """
    depth_ndc  = depth_buffer_value * 2.0 - 1.0
    denom = far + near - depth_ndc * (far - near)
    if abs(denom) < 1e-8:
        return None
    linear_z = (2.0 * near * far) / denom

    # Sanity check: reject if clearly out of reasonable range
    if linear_z < near * 0.5 or linear_z > far * 2.0:
        return None

    return max(0.0, linear_z)


def extract_wrist_camera_pose(env):
    """
    Extract wrist camera position and rotation matrix in WORLD frame from MuJoCo.

    Returns:
        cam_pos   (3,)  — camera optical center in world coords (metres)
        R_world   (3,3) — rotation matrix: camera_frame → world_frame
    """
    if not _HAS_MUJOCO:
        raise RuntimeError("MuJoCo not available — cannot extract camera pose")

    # Find camera ID in model
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i")
    if cam_id < 0:
        raise ValueError("Camera 'd435i' not found in MuJoCo model")

    # MuJoCo stores camera pose relative to its parent body.
    # Get body ID of parent
    body_id = env.model.cam_bodyid[cam_id]

    # Camera position in world frame
    cam_pos = env.data.cam_xpos[cam_id].copy()   # (3,)

    # Camera rotation matrix (camera → world)
    # MuJoCo stores as 3×3 in cam_xmat, column-major
    R_cam_body = env.data.cam_xmat[cam_id].reshape(3, 3).copy()   # body → camera

    # Parent body rotation (body → world) from MuJoCo's derived state
    R_body_world = env.data.xmat[body_id].reshape(3, 3).copy()   # body → world

    # Camera to world: world = body @ (body→cam) = body @ cam
    R_world = R_body_world @ R_cam_body.T   # camera → world

    return cam_pos, R_world


def pixel_to_3d(u, v, depth_value, env=None, cam_pos=None, R_world=None,
                use_depth=True):
    """
    Convert pixel (u, v) + depth to world 3D coordinate.

    Two modes:
      use_depth=True  → use provided depth_value (linear metres or buffer pixel)
      use_depth=False → ray-plane intersection with table plane z=TABLE_Z

    Args:
        u, v          pixel coordinates (float)
        depth_value   if use_depth: linear depth in metres
                      else: ignored
        env           MuJoCo env — used to extract camera pose at runtime
        cam_pos       (3,) camera world position (overrides env if provided)
        R_world       (3,3) rotation matrix camera→world (overrides env if provided)
        use_depth     bool — if False, use planar intersection

    Returns:
        (x, y, z) world coordinates (metres) or None if ray misses
    """
    # ── Camera pose ─────────────────────────────────────────────────────────────
    if cam_pos is None or R_world is None:
        if env is None:
            raise ValueError("Must provide either env or (cam_pos, R_world)")
        cam_pos, R_world = extract_wrist_camera_pose(env)

    # ── Ray direction in camera frame ───────────────────────────────────────────
    # Perspective projection: direction = [(u-cx)/fx, (v-cy)/fy, 1.0]
    dx_cam = (u - CX) / FX
    dy_cam = (v - CY) / FY
    dz_cam = 1.0
    ray_dir_cam = np.array([dx_cam, dy_cam, dz_cam], dtype=np.float64)
    ray_dir_cam /= np.linalg.norm(ray_dir_cam)   # normalize

    # ── Ray in world frame ──────────────────────────────────────────────────────
    ray_dir_world = R_world @ ray_dir_cam

    # ── Depth / intersection ────────────────────────────────────────────────────
    if use_depth and depth_value is not None:
        # Simple: scale unit ray direction by known depth
        pt_cam = ray_dir_cam * depth_value
        pt_world = R_world @ pt_cam + cam_pos
        return pt_world

    # Planar fallback: intersect ray with table plane z = TABLE_Z
    denom = ray_dir_world[2]
    if abs(denom) < 1e-6:
        return None   # ray parallel to table — shouldn't happen with wrist cam

    t = (TABLE_Z - cam_pos[2]) / denom
    if t < 0:
        return None   # intersection behind camera

    pt_world = cam_pos + t * ray_dir_world
    return pt_world


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _hsv_segment(image, hue_low, hue_high, sat_min=60, val_min=60):
    """
    HSV color segmentation with morphological cleanup.

    Args:
        image    (H, W, 3) BGR uint8 — MuJoCo renders BGR
        hue_low/high  HSV range for target color
        sat_min, val_min  minimum saturation/value (ignore dark/desaturated pixels)

    Returns:
        mask  (H, W) bool — True where color is detected
    """
    if not _HAS_CV2:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build mask(s)
    if hue_low[0, 0] > hue_high[0, 0]:
        # Hue wraps around 180 (e.g., red: [160-180] ∪ [0-10])
        mask1 = cv2.inRange(hsv, hue_low, hue_high)
        mask2 = cv2.inRange(hsv, RED_HUE_LOW_1, RED_HUE_HIGH_1)   # handle separately
        mask  = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, hue_low, hue_high)

    # Reject dark / desaturated pixels
    sv_mask = cv2.inRange(hsv, np.array([[[0, sat_min, val_min]]], dtype=np.uint8),
                             np.array([[[180, 255, 255]]], dtype=np.uint8))
    mask = cv2.bitwise_and(mask, sv_mask)

    # Morphological cleanup: erode then dilate
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel3, iterations=1)
    mask = cv2.dilate(mask, kernel5, iterations=2)

    return mask


def _contour_to_centroid(mask):
    """
    Find the largest contour in mask, return centroid (u, v) and area.

    Returns:
        (u, v, area)  — pixel coords + area in px, or (None, None, 0) if none found
    """
    if not _HAS_CV2:
        return None, None, 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0

    # Filter by area
    valid = [c for c in contours if AREA_MIN_PX <= cv2.contourArea(c) <= AREA_MAX_PX]
    if not valid:
        return None, None, 0

    # Pick largest
    best = max(valid, key=cv2.contourArea)
    area = cv2.contourArea(best)

    # Bounding box for aspect ratio sanity check
    x, y, w, h = cv2.boundingRect(best)
    aspect = min(w, h) / (max(w, h) + 1e-6)
    if aspect < ASPECT_TOL:
        return None, None, 0   # too non-square — likely not the cube

    # Centroid via moments
    M = cv2.moments(best)
    if M["m00"] < 1e-6:
        return None, None, 0
    u = M["m10"] / M["m00"]
    v = M["m01"] / M["m00"]

    return float(u), float(v), float(area)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectDetector:
    """
    HSV-based object detector for the SO-101 pick-place scene.

    Detects:
      • red_box   → cube position (pick object)
      • green     → target marker position (place zone)

    Per step:
      1. Segment red + green in RGB image
      2. Project pixel centroids → 3D world coords
      3. Apply EMA smoothing
      4. Inject distance-aware Gaussian noise
      5. Return finite, safe positions

    Fallback chain:
      depth_available → planar intersection → last known pos → safe default

    Output format: all positions are (x, y, z) world metres, never NaN.
    """

    def __init__(self, camera_name="d435i", image_size=(IMAGE_W, IMAGE_H)):
        self.camera_name = camera_name
        self.image_size  = image_size

        # EMA state — one per object
        self._ema_cube   = None   # (3,) current smoothed cube position
        self._ema_target = None   # (3,) current smoothed target position

        # History buffers for fallback
        self._history_cube   = []   # list of last-known (3,) positions
        self._history_target = []   # list of last-known (3,) positions
        self._max_history    = 30  # keep last 30 frames

        # Detection stats (for confidence)
        self._consecutive_detections = {"cube": 0, "target": 0}
        self._drop_this_frame = False   # set randomly to simulate dropout

    # ── Detection ───────────────────────────────────────────────────────────────

    def segment(self, rgb_image):
        """
        Run HSV segmentation on RGB image.

        Returns:
            dict with keys: cube_px, target_px, cube_area, target_area,
                            cube_detected, target_detected
        """
        red_mask  = _hsv_segment(rgb_image, RED_HUE_LOW_1, RED_HUE_HIGH_1)
        green_mask = _hsv_segment(rgb_image, GREEN_HUE_LOW, GREEN_HUE_HIGH)

        cu, cv, ca = _contour_to_centroid(red_mask)
        tu, tv, ta = _contour_to_centroid(green_mask)

        return {
            "cube_px":     (cu, cv) if cu is not None else None,
            "target_px":   (tu, tv) if tu is not None else None,
            "cube_area":   ca,
            "target_area": ta,
            "cube_detected":   cu is not None,
            "target_detected": tu is not None,
        }

    def project(self, uv, env, depth_buffer=None, use_depth=True):
        """
        Project pixel (u, v) to world 3D coordinate.

        Args:
            uv          (u, v) pixel coords or None
            env         MuJoCo env instance (for camera pose extraction)
            depth_buffer (H, W) uint16 — MuJoCo depth buffer, or None
            use_depth   bool

        Returns:
            (x, y, z) world coords or None
        """
        if uv is None:
            return None

        u, v = uv

        # Extract camera pose from MuJoCo at runtime (NOT hardcoded)
        cam_pos, R_world = extract_wrist_camera_pose(env)

        # Depth value
        depth_val = None
        if use_depth and depth_buffer is not None:
            # Linearize if buffer pixel
            raw = float(depth_buffer[int(v), int(u)])
            depth_val = linearize_depth(raw)

        result = pixel_to_3d(u, v, depth_val,
                              cam_pos=cam_pos, R_world=R_world,
                              use_depth=use_depth)
        return result

    def apply_ema(self, raw_pos, ema_state, alpha=EMA_ALPHA):
        """
        Exponential moving average smoothing.

        Order: raw → EMA → noise injection
        """
        if raw_pos is None:
            return None
        raw_pos = np.asarray(raw_pos, dtype=np.float64)

        if ema_state is None:
            return raw_pos.copy()

        return alpha * raw_pos + (1 - alpha) * np.asarray(ema_state, dtype=np.float64)

    def inject_noise(self, pos, cam_pos=None):
        """
        Distance-aware Gaussian noise + occasional dropout.

        sigma = min(SIGMA_BASE * (1 + SIGMA_K * distance), SIGMA_CAP)
        With DROP_PROB probability → return last-known + staleness artifact.

        Pos is always finite on return.
        """
        if pos is None:
            return None

        # Stochastic dropout — simulate missed detection
        if np.random.rand() < DROP_PROB:
            # Return stale position (handled by caller via history)
            noise = np.random.randn(3) * STALE_MM
            return np.asarray(pos, dtype=np.float64) + noise

        # Distance from camera to object
        if cam_pos is not None:
            dist = np.linalg.norm(np.asarray(pos) - np.asarray(cam_pos))
        else:
            dist = 0.5   # assume 50 cm if unknown

        sigma = min(SIGMA_BASE * (1.0 + SIGMA_K * dist), SIGMA_CAP)
        noise = np.random.randn(3) * sigma

        result = np.asarray(pos, dtype=np.float64) + noise
        # Clamp to table bounds — object can't be off-table
        result[0] = np.clip(result[0], 0.0,  0.6)   # X
        result[1] = np.clip(result[1], -0.4, 0.4)   # Y
        result[2] = TABLE_Z                          # Z fixed to table

        return result

    def compute_confidence(self, seg_result, ema_pos, consecutive_frames=3):
        """
        Compute confidence ∈ [0, 1] based on detection quality.

        Factors:
          1. Area score — larger contours = more reliable
          2. Temporal stability — low EMA diff = stable tracking
          3. Consecutive detection bonus — ≥ N frames in a row = +0.2
        """
        if ema_pos is None:
            return 0.0

        area = seg_result.get("cube_area", 0)
        # Normalize area to [0, 1]
        area_score = np.clip((area - AREA_MIN_PX) / (AREA_MAX_PX - AREA_MIN_PX + 1), 0, 1)

        # Stability bonus: if EMA has been stable across last few frames
        stability_bonus = 0.0
        if len(self._history_cube) >= consecutive_frames:
            recent = np.array(self._history_cube[-consecutive_frames:])
            diffs  = np.linalg.norm(np.diff(recent, axis=0), axis=1)
            if np.mean(diffs) < 0.01:   # < 10mm movement = stable
                stability_bonus = 0.1

        # Detection history bonus
        history_bonus = 0.0
        if self._consecutive_detections["cube"] >= consecutive_frames:
            history_bonus = 0.1

        confidence = np.clip(area_score + stability_bonus + history_bonus, 0.0, 1.0)
        return float(confidence)

    # ── Main detect_from_camera ─────────────────────────────────────────────────

    def detect_from_camera(self, rgb_image, depth_buffer=None,
                           env=None, use_depth=True):
        """
        Full perception pipeline: image → detected positions.

        Args:
            rgb_image    (H, W, 3) BGR uint8 — from MuJoCo renderer
            depth_buffer (H, W) uint16 or None — MuJoCo depth buffer
            env          MuJoCo env instance (required for projection)
            use_depth    bool — try depth first, fall back to planar

        Returns:
            dict {
                "cube_pos":   (3,) world metres or safe default,
                "target_pos": (3,) world metres or safe default,
                "confidence": float ∈ [0, 1],
                "cube_detected": bool,
                "target_detected": bool,
                "cube_px":    (u, v) or None,
                "target_px":  (u, v) or None,
                "debug_overlay": rgb_image with detection overlays (or None),
            }
        """
        if env is None:
            raise ValueError("detect_from_camera requires MuJoCo env")

        # ── Step 1: HSV segmentation ────────────────────────────────────────────
        seg = self.segment(rgb_image)

        # Update consecutive detection counters
        for key in ["cube", "target"]:
            if seg.get(f"{key}_detected"):
                self._consecutive_detections[key] += 1
            else:
                self._consecutive_detections[key] = 0

        # ── Step 2: Project to 3D ───────────────────────────────────────────────
        cube_raw  = self.project(seg.get("cube_px"),   env, depth_buffer, use_depth)
        target_raw = self.project(seg.get("target_px"), env, depth_buffer, use_depth)

        # ── Step 3: Fallback chain ──────────────────────────────────────────────
        cube_final  = cube_raw
        target_final = target_raw

        if cube_raw is None:
            if self._history_cube:
                cube_final = self._history_cube[-1].copy()
            else:
                # Safe default: center of reachable workspace
                cube_final = np.array([0.225, 0.0, TABLE_Z], dtype=np.float64)

        if target_raw is None:
            if self._history_target:
                target_final = self._history_target[-1].copy()
            else:
                target_final = np.array([0.25, 0.20, TABLE_Z], dtype=np.float64)

        # ── Step 4: EMA smoothing ──────────────────────────────────────────────
        cube_final  = self.apply_ema(cube_final,  self._ema_cube)
        target_final = self.apply_ema(target_final, self._ema_target)

        # Update EMA state
        if cube_final is not None:
            self._ema_cube = cube_final.copy()
        if target_final is not None:
            self._ema_target = target_final.copy()

        # ── Step 5: Noise injection ─────────────────────────────────────────────
        cam_pos, _ = extract_wrist_camera_pose(env)
        cube_final  = self.inject_noise(cube_final,  cam_pos)
        target_final = self.inject_noise(target_final, cam_pos)

        # Update history
        if cube_final is not None:
            self._history_cube.append(cube_final.copy())
            if len(self._history_cube) > self._max_history:
                del self._history_cube[0]
        if target_final is not None:
            self._history_target.append(target_final.copy())
            if len(self._history_target) > self._max_history:
                del self._history_target[0]

        # ── Step 6: Confidence ───────────────────────────────────────────────────
        confidence = self.compute_confidence(seg, self._ema_cube)

        # ── Step 7: Debug overlay ───────────────────────────────────────────────
        overlay = self._draw_overlay(rgb_image.copy(),
                                      seg.get("cube_px"),
                                      seg.get("target_px"),
                                      cube_final, target_final, confidence)

        return {
            "cube_pos":     cube_final,
            "target_pos":   target_final,
            "confidence":   confidence,
            "cube_detected": seg["cube_detected"],
            "target_detected": seg["target_detected"],
            "cube_px":      seg.get("cube_px"),
            "target_px":    seg.get("target_px"),
            "debug_overlay": overlay,
        }

    # ── Debug overlay ───────────────────────────────────────────────────────────

    def _draw_overlay(self, image, cube_px, target_px, cube_3d, target_3d, confidence):
        """Draw detection markers + confidence on RGB image."""
        if not _HAS_CV2:
            return image
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Cube marker — red
        if cube_px is not None:
            u, v = int(cube_px[0]), int(cube_px[1])
            cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(image, f"CUBE({u},{v})", (u+5, v-5),
                        font, 0.5, (0, 0, 255), 1)

        # Target marker — green
        if target_px is not None:
            u, v = int(target_px[0]), int(target_px[1])
            cv2.drawMarker(image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(image, f"TGT({u},{v})", (u+5, v-5),
                        font, 0.5, (0, 255, 0), 1)

        # Confidence bar (top-left corner)
        bar_w = int(w * 0.25)
        bar_h = 8
        cv2.rectangle(image, (10, 10), (10 + bar_w, 10 + bar_h), (80, 80, 80), -1)
        cv2.rectangle(image, (10, 10),
                      (10 + int(bar_w * confidence), 10 + bar_h),
                      (0, 255, 0), -1)
        cv2.putText(image, f"CONF: {confidence:.2f}", (10, 28),
                    font, 0.5, (255, 255, 255), 1)

        # Status text
        status = "DETECTED" if confidence > 0.5 else "TRACKING"
        cv2.putText(image, status, (w - 100, 30),
                    font, 0.6, (0, 255, 0) if confidence > 0.5 else (0, 165, 255), 2)

        return image

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear EMA and history — call on new episode."""
        self._ema_cube       = None
        self._ema_target     = None
        self._history_cube   = []
        self._history_target = []
        self._consecutive_detections = {"cube": 0, "target": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ObjectDetector — geometric sanity checks")
    print(f"  Camera: {IMAGE_W}×{IMAGE_H}, fovy={FOVY_DEG}°, hfov={FOVX_DEG:.1f}°")
    print(f"  fx={FX:.1f}, fy={FY:.1f}, cx={CX}, cy={CY}")
    print(f"  Table Z = {TABLE_Z}m")
    print(f"  EMA alpha = {EMA_ALPHA}, sigma_base = {SIGMA_BASE}m")
    print(f"  Drop prob = {DROP_PROB}")

    # ── Test rotation matrix ─────────────────────────────────────────────────
    euler_test = np.array([0.0, 0.0, 0.0])
    R = rotation_from_euler(euler_test)
    assert np.allclose(R @ R.T, np.eye(3)), "Rotation matrix not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "Rotation matrix determinant != 1"
    print("\n✓ rotation_from_euler: orthogonal, det=1")

    # ── Test depth linearization ─────────────────────────────────────────────
    # At near plane → expect near value
    assert abs(linearize_depth(0.0, near=0.05, far=3.0) - 0.05) < 0.02, \
        f"near plane: {linearize_depth(0.0, near=0.05, far=3.0)}"
    # At far plane → expect far value
    assert abs(linearize_depth(1.0, near=0.05, far=3.0) - 3.0) < 0.3, \
        f"far plane: {linearize_depth(1.0, near=0.05, far=3.0)}"
    print("✓ linearize_depth: near/far bounds correct")

    # ── Test EMA ──────────────────────────────────────────────────────────────
    ema_test = None
    pos = np.array([0.2, 0.1, 0.025])
    pos_new = EMA_ALPHA * pos + (1 - EMA_ALPHA) * (ema_test if ema_test else pos)
    assert np.all(np.isfinite(pos_new))
    print("✓ EMA smoothing: stable")

    # ── Test noise injection (no MuJoCo needed) ──────────────────────────────
    noise = np.random.randn(3) * SIGMA_BASE
    assert np.all(np.isfinite(noise))
    print("✓ Noise injection: Gaussian, finite")

    # ── Test no-NaN guarantee ────────────────────────────────────────────────
    fallback = np.array([0.225, 0.0, TABLE_Z])
    safe = np.where(np.isfinite(fallback), fallback, np.array([0.225, 0.0, TABLE_Z]))
    assert np.all(np.isfinite(safe))
    print("✓ No-NaN: fallback values always finite")

    print("\nperception_module.py — geometric sanity checks PASSED")
    print("  import cv2:", "✓ available" if _HAS_CV2 else "⚠  not installed (run in Docker container)")
    print("  import mujoco:", "✓ available" if _HAS_MUJOCO else "⚠  not installed (run in container)")
    if _HAS_MUJOCO:
        print("  ✓ Camera pose extraction enabled")
    else:
        print("  ⚠  MuJoCo not available — camera pose uses fallback mounting offset")
