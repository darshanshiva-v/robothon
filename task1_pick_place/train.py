#!/usr/bin/env python3
"""
train.py — Behavior Cloning for SO101 pick-and-place.
MLP: Input(13) → 256 → ReLU → 256 → ReLU → 128 → ReLU → Output(6)

Changes from original:
- Wider layers (256-256-128 vs 128-128-64)
- L1 loss instead of MSE (more robust to outlier waypoints)
- Cosine annealing with warm restarts for better convergence
- Normalizes observations per-dimension (zero-mean, unit-variance)
- More epochs (200 vs 100)
- Gradient clipping (max_norm=1.0)
"""

import os
import pickle
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ── Normalization ─────────────────────────────────────────────────────────────

class ObsNorm(nn.Module):
    """Learnable observation normalization — per-dimension mean/std."""
    def __init__(self, dim=13):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x):
        return x * self.std + self.mean


class PolicyMLP(nn.Module):
    """
    Behavior cloning MLP with observation normalization.
    Input:  observation (13,)  — 7 joint angles + 3 cube pos + 3 target pos
    Output: joint position command (6,)
    """
    def __init__(self, in_dim=13, hidden1=256, hidden2=256, hidden3=128, out_dim=6):
        super().__init__()
        self.obs_norm = ObsNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, out_dim),
        )

    def forward(self, x):
        x_norm = self.obs_norm.normalize(x)
        return self.net(x_norm)

    def normalize_dataset(self, obs, act):
        """Compute normalization stats from training data."""
        mean = torch.from_numpy(obs.mean(axis=0)).float()
        std  = torch.from_numpy(obs.std(axis=0) + 1e-8).float()
        self.obs_norm.mean.copy_(mean)
        self.obs_norm.std.copy_(std)


# ── Training ─────────────────────────────────────────────────────────────────

def load_demos(path):
    with open(path, "rb") as f:
        transitions = pickle.load(f)
    obs_batch  = np.array([t[0] for t in transitions], dtype=np.float32)
    act_batch  = np.array([t[1] for t in transitions], dtype=np.float32)
    return obs_batch, act_batch


def train(obs, act, epochs=200, batch_size=64, lr=5e-4, device="cpu"):
    dataset = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(act),
    )
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = PolicyMLP().to(device)

    # Normalize observations using training data stats
    policy.normalize_dataset(obs, act)

    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()  # L1 + L2 blend — robust to outliers

    best_loss   = float("inf")
    best_state  = None
    patience    = 0
    max_patience = 30

    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            pred = policy(batch_obs)
            loss = criterion(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = copy.deepcopy(policy.state_dict())
            patience   = 0
        else:
            patience += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best
    policy.load_state_dict(best_state)
    return policy, best_loss


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demos_path = os.path.join(script_dir, "demos.pkl")
    save_path  = os.path.join(script_dir, "policy.pt")
    config_path = os.path.join(script_dir, "policy_scripted_config.json")

    print("Loading demos...")
    obs, act = load_demos(demos_path)
    print(f"  Observations: {obs.shape}, Actions: {act.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    policy, final_loss = train(obs, act, epochs=200, batch_size=64, lr=5e-4, device=device)

    checkpoint = {
        "model_type": "scripted_bc_mlp",
        "policy_state_dict": policy.state_dict(),
        "obs_dim": 13,
        "action_dim": 6,
        "hidden_dims": [256, 256, 128],
        "normalization": {
            "mean": policy.obs_norm.mean.detach().cpu(),
            "std": policy.obs_norm.std.detach().cpu(),
        },
        "training": {
            "best_loss": float(final_loss),
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 5e-4,
            "dataset_path": demos_path,
        },
    }
    torch.save(checkpoint, save_path)
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_type": checkpoint["model_type"],
                "obs_dim": checkpoint["obs_dim"],
                "action_dim": checkpoint["action_dim"],
                "hidden_dims": checkpoint["hidden_dims"],
                "best_loss": checkpoint["training"]["best_loss"],
                "dataset_path": demos_path,
            },
            f,
            indent=2,
        )
    print(f"\nBest loss: {final_loss:.6f}")
    print(f"Saved policy → {save_path}")
    print(f"Saved config → {config_path}")


if __name__ == "__main__":
    main()
