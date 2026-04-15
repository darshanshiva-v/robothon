#!/usr/bin/env python3
"""
train_policy.py — PyTorch behavior-cloning policy for SO-101 pick-and-place.

Architecture:
    Input (39,)  — 3-frame stacked observation (3 × 13 raw obs)
    → Linear(39, 256) → LayerNorm(256) → ReLU → Dropout(0.05)
    → Linear(256, 256) → LayerNorm(256) → ReLU → Dropout(0.05)
    → Linear(256, 128) → ReLU
    → Linear(128, 7)
    Output: [arm_joints (6), gripper_sigmoid (1)]

Loss:
    joint_loss  = SmoothL1Loss(output[:, 0:6], target[:, 0:6])
    gripper_loss = BCEWithLogitsLoss(sigmoid(output[:, 6]), target[:, 6])
    total_loss  = 0.9 * joint_loss + 0.1 * gripper_loss

Data: reads teleop_data.hdf5 from teleop_data_collector.py
Training: AdamW(lr=5e-4), CosineAnnealingWarmRestarts, 200 epochs, early stopping
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py

# ── Hyperparameters ────────────────────────────────────────────────────────────

OBS_DIM_STACKED = 39          # 3 frames × 13 raw obs
ACTION_DIM = 7                # 6 arm joints + 1 binary gripper
N_ARM_JOINTS = 6
N_FRAMES = 3                   # temporal frame stack size

HIDDEN_DIMS = [256, 256, 128]  # MLP hidden layers

EPOCHS = 200
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 30                 # early stopping patience

LOSS_JOINT_WEIGHT = 0.9
LOSS_GRIPPER_WEIGHT = 0.1

TRAIN_RATIO = 0.8             # 80% train, 20% val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HDF5_PATH = os.path.join(SCRIPT_DIR, "teleop_data.hdf5")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "policy_teleop.pt")
CONFIG_OUT = os.path.join(SCRIPT_DIR, "policy_teleop_config.json")


# ── Observation Normalization ─────────────────────────────────────────────────

class ObsNorm(nn.Module):
    """
    Learnable per-dimension mean/std for observations.
    Fitted on training data before training starts.
    """
    def __init__(self, dim=OBS_DIM_STACKED):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std",  torch.ones(dim))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)


# ── Policy Network ─────────────────────────────────────────────────────────────

class PolicyMLP(nn.Module):
    """
    Behavior-cloning MLP for pick-and-place.

    Input:  stacked observation (39,) — 3 frames × 13 raw dims
    Output: 7-dim action
              indices [0:6]  — arm joint targets (MSE/SmoothL1 supervised)
              index  [6]    — gripper sigmoid logit (BCE supervised)

    The 6 arm outputs are unconstrained (linear).
    The gripper output is treated as a logit → sigmoid → BCEWithLogitsLoss.
    """
    def __init__(
        self,
        in_dim=OBS_DIM_STACKED,
        hidden=[256, 256, 128],
        out_dim=ACTION_DIM,
        dropout=0.05,
    ):
        super().__init__()
        self.obs_norm = ObsNorm(in_dim)

        layers = []
        prev_dim = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, out_dim)

        # Initialize output head to near-zero for stable early training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x_norm = self.obs_norm(x)
        features = self.backbone(x_norm)
        return self.head(features)

    def normalize_dataset(self, obs_np):
        """
        Compute normalization stats from training observations.
        Call once after loading data, before training.
        """
        mean = torch.from_numpy(obs_np.mean(axis=0)).float()
        std  = torch.from_numpy(obs_np.std(axis=0) + 1e-8).float()
        self.obs_norm.mean.copy_(mean)
        self.obs_norm.std.copy_(std)


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_hdf5(path):
    """
    Load demonstration data from HDF5 collected by teleop_data_collector.py.

    Returns:
        obs_np  (N, 13) float32  — raw observations (13-dim per frame)
        act_np  (N, 7)  float32  — actions [joints (6) + gripper_binary (1)]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run teleop_data_collector.py first to collect demonstrations."
        )

    with h5py.File(path, "r") as f:
        obs_raw = f["episodes/obs"][:]          # (N, 13)
        act_raw = f["episodes/act"][:]           # (N, 7)

    print(f"Loaded {obs_raw.shape[0]} samples from {path}")
    return obs_raw, act_raw


def build_stacked_obs(raw_obs, n_frames=3):
    """
    Reconstruct 3-frame stacked observations from raw (N, 13) dataset.

    For frame i, the stacked input is:
        [obs[i-2], obs[i-1], obs[i]]  (chronological order, padded at start)

    Returns (N, 39) array.
    """
    n = raw_obs.shape[0]
    stacked = np.zeros((n, n_frames * raw_obs.shape[1]), dtype=np.float32)

    for i in range(n):
        frames = []
        for offset in range(n_frames - 1, -1, -1):
            idx = max(0, i - offset)
            frames.append(raw_obs[idx])
        stacked[i] = np.concatenate(frames)

    return stacked


def prepare_data(raw_obs, raw_act, train_ratio=TRAIN_RATIO):
    """
    Build stacked observations, split into train/val, wrap in DataLoaders.

    raw_obs: (N, 13) raw observations from HDF5
    raw_act: (N, 7)  actions from HDF5

    Returns: (train_loader, val_loader, n_train, n_val, obs_mean, obs_std)
    """
    # Build stacked frames
    obs_stacked = build_stacked_obs(raw_obs, n_frames=N_FRAMES)

    # Train/val split
    n = obs_stacked.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * train_ratio)
    train_idx, val_idx = idx[:split], idx[split:]

    train_obs = obs_stacked[train_idx]
    train_act = raw_act[train_idx]
    val_obs   = obs_stacked[val_idx]
    val_act   = raw_act[val_idx]

    # Convert to tensors
    train_obs_t = torch.from_numpy(train_obs).float()
    train_act_t = torch.from_numpy(train_act).float()
    val_obs_t   = torch.from_numpy(val_obs).float()
    val_act_t   = torch.from_numpy(val_act).float()

    train_loader = DataLoader(
        TensorDataset(train_obs_t, train_act_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_obs_t, val_act_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print(f"  Train: {train_obs_t.shape[0]} samples")
    print(f"  Val:   {val_obs_t.shape[0]} samples")

    return train_loader, val_loader


# ── Training Loop ──────────────────────────────────────────────────────────────

def compute_loss(pred, target):
    """
    Combined loss for joint positions + gripper.

    pred[:, 0:6]   — arm joint targets (SmoothL1)
    pred[:, 6]      — gripper logit (BCEWithLogitsLoss)
    target[:, 6]   — gripper binary {0, 1}

    Gripper targets from teleop_data_collector.py are binary {0, 1}.
    """
    # Joint loss — SmoothL1 is robust to outlier waypoints
    joint_pred  = pred[:, 0:N_ARM_JOINTS]
    joint_target = target[:, 0:N_ARM_JOINTS]
    joint_loss = nn.functional.smooth_l1_loss(joint_pred, joint_target)

    # Gripper loss — BCEWithLogitsLoss expects raw logits (no sigmoid in forward)
    # This is numerically stable: internally does log(sigmoid(logit)) etc.
    gripper_logits = pred[:, N_ARM_JOINTS]
    gripper_target = target[:, N_ARM_JOINTS]    # already binary {0, 1}
    gripper_loss = nn.functional.binary_cross_entropy_with_logits(
        gripper_logits, gripper_target
    )

    total = LOSS_JOINT_WEIGHT * joint_loss + LOSS_GRIPPER_WEIGHT * gripper_loss
    return total, joint_loss.detach(), gripper_loss.detach()


def train_epoch(policy, loader, optimizer, device):
    policy.train()
    total_loss = 0.0
    total_joint = 0.0
    total_gripper = 0.0
    n_batches = 0

    for batch_obs, batch_act in loader:
        batch_obs = batch_obs.to(device)
        batch_act = batch_act.to(device)

        pred = policy(batch_obs)
        loss, j_loss, g_loss = compute_loss(pred, batch_act)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss   += loss.item()
        total_joint  += j_loss.item()
        total_gripper += g_loss.item()
        n_batches    += 1

    return total_loss / n_batches, total_joint / n_batches, total_gripper / n_batches


@torch.no_grad()
def validate(policy, loader, device):
    policy.eval()
    total_loss = 0.0
    total_joint = 0.0
    total_gripper = 0.0
    n_batches = 0

    for batch_obs, batch_act in loader:
        batch_obs = batch_obs.to(device)
        batch_act = batch_act.to(device)

        pred = policy(batch_obs)
        loss, j_loss, g_loss = compute_loss(pred, batch_act)

        total_loss   += loss.item()
        total_joint  += j_loss.item()
        total_gripper += g_loss.item()
        n_batches    += 1

    return total_loss / n_batches, total_joint / n_batches, total_gripper / n_batches


def run_training(policy, train_loader, val_loader, epochs=EPOCHS, device=DEVICE):
    optimizer = optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    print(f"\nTraining on {device.upper()} — {epochs} epochs max, patience={PATIENCE}")
    print(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | {'Joint':>8} | {'Gripper':>8} | LR")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        train_loss, train_j, train_g = train_epoch(policy, train_loader, optimizer, device)
        val_loss,   val_j,   val_g   = validate(policy, val_loader, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]

        # Log every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"{epoch:6d} | {train_loss:10.4f} | {val_loss:10.4f} | "
                f"{train_j:8.4f} | {train_g:8.4f} | {lr:.2e}"
            )

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(policy.state_dict())
            patience_cnt  = 0
        else:
            patience_cnt += 1

        # Early stopping
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best
    policy.load_state_dict(best_state)
    return policy, best_val_loss


# ── Main ───────────────────────────────────────────────────────────────────────

def _load_pickle_demos(path):
    """Load demos from pickle (scripted collection via collect_demos.py)."""
    import pickle
    with open(path, "rb") as f:
        transitions = pickle.load(f)
    obs_np = np.array([t[0] for t in transitions], dtype=np.float32)
    # Pickle demos: 6-dim actions, no gripper dimension.
    # Pad gripper to 7-dim: assume gripper was always open (0.0) in scripted demos.
    act_np = np.full((len(transitions), 7), 0.0, dtype=np.float32)
    act_np[:, :6] = np.array([t[1] for t in transitions], dtype=np.float32)
    return obs_np, act_np


def main():
    print("=" * 60)
    print("SO-101 Behavior Cloning — Training")
    print("=" * 60)

    data_path = HDF5_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Teleop dataset not found at {data_path}.\n"
            "Run `python3 task1_pick_place/teleop_data_collector.py` first."
        )
    print("\nLoading data from:", data_path)
    raw_obs, raw_act = load_hdf5(data_path)

    # 2. Prepare train/val
    print("\nBuilding stacked observations (3-frame)...")
    train_loader, val_loader = prepare_data(raw_obs, raw_act)

    # 3. Init policy
    policy = PolicyMLP(
        in_dim=OBS_DIM_STACKED,
        hidden=HIDDEN_DIMS,
        out_dim=ACTION_DIM,
    ).to(DEVICE)

    # 4. Normalize observations using training stats
    print("Fitting observation normalization...")
    policy.normalize_dataset(train_loader.dataset.tensors[0].numpy())

    # 5. Train
    policy, best_loss = run_training(policy, train_loader, val_loader, device=DEVICE)

    # 6. Save
    checkpoint = {
        "model_type": "teleop_bc_mlp",
        "policy_state_dict": policy.state_dict(),
        "obs_dim_stacked": OBS_DIM_STACKED,
        "action_dim": ACTION_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "normalization": {
            "mean": policy.obs_norm.mean.detach().cpu(),
            "std": policy.obs_norm.std.detach().cpu(),
        },
        "training": {
            "best_val_loss": float(best_loss),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "train_ratio": TRAIN_RATIO,
            "dataset_path": data_path,
        },
    }
    torch.save(checkpoint, MODEL_OUT)
    print(f"\nBest val loss: {best_loss:.6f}")
    print(f"Policy saved:  {MODEL_OUT}")

    # 7. Save config for evaluation
    config = {
        "obs_dim_stacked": OBS_DIM_STACKED,
        "action_dim":      ACTION_DIM,
        "n_arm_joints":    N_ARM_JOINTS,
        "hidden_dims":     HIDDEN_DIMS,
        "model_type":      checkpoint["model_type"],
        "epochs_trained":  EPOCHS,
        "best_val_loss":   float(best_loss),
        "device":          DEVICE,
        "hdf5_path":       data_path,
        "train_ratio":     TRAIN_RATIO,
        "lr":              LR,
        "batch_size":      BATCH_SIZE,
    }
    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved:   {CONFIG_OUT}")

    print("\ntrain_policy.py — DONE")


if __name__ == "__main__":
    main()
