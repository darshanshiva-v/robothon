import os
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from run_robot import EXPERT_PLAN, MODEL_PATH, POSES, PolicyNet, SO101PickPlaceDemo


OUT_PATH = "models/so101_pick_place_policy.pt"


@dataclass
class SampleBatch:
    obs: list
    act: list


def collect_demonstrations(episodes: int = 30):
    demo = SO101PickPlaceDemo(controller_mode="expert", policy_path=OUT_PATH)
    batch = SampleBatch(obs=[], act=[])

    for _ in range(episodes):
        demo.reset()
        max_steps = 700
        for _ in range(max_steps):
            vision = demo.detect_scene()
            obs = demo.observation(vision)
            act = demo.expert_action(vision)
            batch.obs.append(obs.astype(np.float32))
            batch.act.append(act.astype(np.float32))

            demo.target_ctrl = act.copy()
            demo.data.ctrl[:] = demo.data.ctrl + 0.08 * (demo.target_ctrl - demo.data.ctrl)
            demo.maybe_attach_or_release()
            for _ in range(4):
                demo.apply_object_physics_assist()
                mujoco.mj_step(demo.model, demo.data)
                mujoco.mj_forward(demo.model, demo.data)

            if demo.cycle_complete:
                break

    return np.asarray(batch.obs), np.asarray(batch.act)


def train_model(obs, act):
    dataset = TensorDataset(torch.from_numpy(obs), torch.from_numpy(act))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = PolicyNet(obs.shape[1], act.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(120):
        total = 0.0
        count = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1:03d} loss {total / max(count, 1):.6f}")

    return model


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    obs, act = collect_demonstrations()
    print(f"collected {len(obs)} samples")
    model = train_model(obs, act)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs.shape[1],
            "act_dim": act.shape[1],
        },
        OUT_PATH,
    )
    print(f"saved learned policy to {OUT_PATH}")


if __name__ == "__main__":
    main()
