#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained policy on 20 pick-and-place trials.
Prints per-trial accuracy table and summary statistics.

Usage:
    python evaluate.py                    # fixed target (as trained)
    python evaluate.py --random           # randomized cube+target (tests generalization)
"""

import os
import argparse
import numpy as np
import torch

try:
    from .environment import PickPlaceEnv
    from .train import PolicyMLP
except ImportError:
    from environment import PickPlaceEnv
    from train import PolicyMLP


def load_policy(path, device="cpu"):
    policy = PolicyMLP().to(device)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "policy_state_dict" in payload:
            state_dict = payload["policy_state_dict"]
        else:
            state_dict = payload
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint at {path}. "
            "The saved weights do not match the current PolicyMLP architecture. "
            "Retrain with `python3 task1_pick_place/train.py` to regenerate policy.pt."
        ) from exc
    policy.eval()
    return policy


def evaluate(policy, n_trials=20, device="cpu", verbose=True, randomize=False):
    env    = PickPlaceEnv()
    results = []

    for trial in range(1, n_trials + 1):
        obs = env.reset(randomize_cube=randomize, randomize_target=randomize)
        done   = False
        steps  = 0

        with torch.no_grad():
            while not done and steps < 400:
                x  = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = policy(x).cpu().numpy().flatten()
                obs, _, done, info = env.step(action)
                steps += 1

        cube   = info["cube_pos"]
        target = info["target"]
        error  = float(np.linalg.norm(cube[:2] - target[:2]))
        error_mm = error * 1000
        passed  = error_mm < 30.0   # 30mm threshold
        status  = "PASS" if passed else "FAIL"

        results.append({
            "trial": trial,
            "target_x": target[0],
            "target_y": target[1],
            "placed_x": cube[0],
            "placed_y": cube[1],
            "error_mm": error_mm,
            "status": status,
            "steps": steps,
        })

        if verbose:
            print(f"Trial {trial:2d} | Target: ({target[0]:.3f}, {target[1]:.3f}) "
                  f"| Placed: ({cube[0]:.3f}, {cube[1]:.3f}) "
                  f"| Error: {error_mm:.1f}mm | {status}")

    return results


def print_summary(results):
    errors  = [r["error_mm"] for r in results]
    passes  = sum(1 for r in results if r["status"] == "PASS")
    n       = len(results)

    print(f"\n{'='*60}")
    print(f"  Success Rate: {passes}/{n}  |  "
          f"Mean Error: {np.mean(errors):.1f}mm  |  "
          f"Best: {np.min(errors):.1f}mm")
    print(f"{'='*60}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(script_dir, "policy.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Randomize cube and target during evaluation.")
    parser.add_argument("--trials", type=int, default=20, help="Number of evaluation episodes.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device.upper()}")

    policy = load_policy(policy_path, device=device)

    results = evaluate(
        policy,
        n_trials=args.trials,
        device=device,
        verbose=True,
        randomize=args.random,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
