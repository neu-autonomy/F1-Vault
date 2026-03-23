# """
# training.py

# Training entrypoint for the F1-Vault TerrainEncoder.

# Supports multiple simulators:
#   - dphysics  : repo differentiable physics (if deps available)
#   - kinematic : built-in differentiable unicycle model (always available)
#   - none      : supervised/regularization only
# """

# from __future__ import annotations

# import argparse
# import json
# import math
# import os
# import random
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional

# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset

# from functools import partial

# from f1_vault.models import TerrainEncoder
# from f1_vault.losses import LossWeights, PhysicsInformedTerrainLoss


# Tensor = torch.Tensor
# def collate_chunks_with_scales(chunks, control_v_scale: float, control_w_scale: float):
#     items = [c.to_tensors() for c in chunks]
#     bev_map = torch.stack([it["bev_map"] for it in items], dim=0)
#     initial_pose = torch.stack([it["initial_pose"] for it in items], dim=0)
#     actions = torch.stack([it["actions"] for it in items], dim=0)
#     gt_traj = torch.stack([it["gt_trajectory"] for it in items], dim=0)

#     # IMPORTANT: keep raw dataset actions (throttle, steering) in [-1,1]
#     # Let DPhysics convert internally via cfg.control_mode="throttle_steer"
#     return {"bev_map": bev_map, "initial_pose": initial_pose, "actions": actions, "gt_trajectory": gt_traj}

# def seed_everything(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def pick_device(device: str = "auto") -> torch.device:
#     if device != "auto":
#         return torch.device(device)
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def quat_wxyz_from_yaw(yaw: Tensor) -> Tensor:
#     half = 0.5 * yaw
#     qw = torch.cos(half)
#     qx = torch.zeros_like(qw)
#     qy = torch.zeros_like(qw)
#     qz = torch.sin(half)
#     return torch.stack([qw, qx, qy, qz], dim=-1)


# class KinematicUnicycleSimulator(nn.Module):
#     """
#     Differentiable unicycle model fallback.
#     controls: (B,T,2) with (v,w)
#     z is tied to mean(z_grid) to permit gradients into terrain predictions.
#     """

#     def __init__(self, dt: float = 0.1):
#         super().__init__()
#         self.dt = float(dt)

#     def forward(self, z_grid: Tensor, controls: Tensor, state=None, friction=None) -> Dict[str, Tensor]:
#         B, T, _ = controls.shape
#         v = controls[..., 0]
#         w = controls[..., 1]

#         if state is not None and isinstance(state, (tuple, list)) and len(state) >= 3:
#             x0 = state[0]  # (B,3)
#             R0 = state[2]  # (B,3,3)
#             x = x0[:, 0]
#             y = x0[:, 1]
#             yaw = torch.atan2(R0[:, 1, 0], R0[:, 0, 0] + 1e-8)
#         else:
#             x = torch.zeros(B, device=z_grid.device, dtype=z_grid.dtype)
#             y = torch.zeros_like(x)
#             yaw = torch.zeros_like(x)

#         z0 = z_grid.mean(dim=(-2, -1))  # (B,)

#         xs, ys, zs, yaws = [], [], [], []
#         for t in range(T):
#             xs.append(x)
#             ys.append(y)
#             zs.append(z0)
#             yaws.append(yaw)

#             x = x + v[:, t] * torch.cos(yaw) * self.dt
#             y = y + v[:, t] * torch.sin(yaw) * self.dt
#             yaw = yaw + w[:, t] * self.dt

#         positions = torch.stack(
#             [torch.stack(xs, dim=1), torch.stack(ys, dim=1), torch.stack(zs, dim=1)],
#             dim=-1,
#         )
#         yaws = torch.stack(yaws, dim=1)
#         return {"positions": positions, "yaws": yaws}


# class SyntheticTerrainTrajectoryDataset(Dataset):
#     """
#     Minimal synthetic dataset for pipeline testing.
#     """

#     def __init__(self, n_samples: int = 512, H: int = 26, W: int = 26, T: int = 10, dt: float = 0.1):
#         self.n_samples = int(n_samples)
#         self.H, self.W, self.T = int(H), int(W), int(T)
#         self.dt = float(dt)
#         self.sim = KinematicUnicycleSimulator(dt=dt)

#     def __len__(self) -> int:
#         return self.n_samples

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         bev = torch.randn(1, self.H, self.W) * 0.1
#         terrain_gt = {
#             "height": bev[0],
#             "stiffness": torch.ones(self.H, self.W) * 200,
#             "damping": torch.ones(self.H, self.W) * 5,
#             "friction": torch.ones(self.H, self.W) * 1.0,
#         }
#         controls = torch.randn(self.T, 2) * 0.2

#         sim_out = self.sim(z_grid=terrain_gt["height"].unsqueeze(0), controls=controls.unsqueeze(0))
#         pos = sim_out["positions"][0]  # (T,3)
#         yaw = sim_out["yaws"][0]       # (T,)

#         quat = quat_wxyz_from_yaw(yaw)
#         gt_pose = torch.cat([pos, quat], dim=-1)

#         return {
#             "bev_map": bev,
#             "initial_pose": torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
#             "controls": controls,
#             "gt_trajectory": gt_pose,
#             "terrain_gt": terrain_gt,
#         }


# def collate_dict_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     out: Dict[str, Any] = {}
#     out["bev_map"] = torch.stack([b["bev_map"] for b in batch], dim=0)
#     out["initial_pose"] = torch.stack([b["initial_pose"] for b in batch], dim=0)
#     out["controls"] = torch.stack([b["controls"] for b in batch], dim=0)
#     out["gt_trajectory"] = torch.stack([b["gt_trajectory"] for b in batch], dim=0)

#     if "terrain_gt" in batch[0]:
#         keys = batch[0]["terrain_gt"].keys()
#         out["terrain_gt"] = {k: torch.stack([b["terrain_gt"][k] for b in batch], dim=0) for k in keys}
#     return out


# def try_make_dphysics_simulator(dt: float) -> Optional[Callable[..., Any]]:
#     """
#     Attempt to build repo DPhysics simulator. Returns None if deps are missing.
#     """
#     try:
#         from f1_vault.physics.physicscfg import DPhysConfig
#         from f1_vault.physics.physics import DPhysics

#         cfg = DPhysConfig()
#         cfg.dt = float(dt)
#         cfg.traj_sim_time = float(dt) * 10.0
#         cfg.use_odeint = False

#         # ensure throttle/steer mode is used
#         cfg.control_mode = "throttle_steer"
#         cfg.throttle_to_v = 3.0
#         cfg.steer_to_delta = 0.488
#         cfg.wheelbase = 0.33
#         cfg.omega_clip = 6.0

#         sim = DPhysics(dphys_cfg=cfg, device=str(pick_device("auto")))
#         sim.eval()
#         return sim
#     except Exception as e:
#         print(f"[WARN] DPhysics unavailable. Reason: {type(e).__name__}: {e}")
#         return None


# def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
#     name = name.lower()
#     if name == "adam":
#         return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     if name == "adamw":
#         return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     raise ValueError(f"Unknown optimizer: {name}")


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Train F1-Vault TerrainEncoder")

#     p.add_argument("--hdf5_path", type=str, default="", help="Path to HDF5 dataset. If empty, use synthetic data.")
#     p.add_argument("--batch_size", type=int, default=8)
#     p.add_argument("--num_workers", type=int, default=0)
#     p.add_argument("--epochs", type=int, default=20)
#     p.add_argument("--seed", type=int, default=42)

#     p.add_argument("--input_channels", type=int, default=1)
#     p.add_argument("--hidden_dims", type=int, nargs="*", default=[32, 64, 64])
#     p.add_argument("--min_stiffness", type=float, default=100.0)
#     p.add_argument("--height_scale", type=float, default=0.5)
#     p.add_argument("--weights", type=str, default="")

#     p.add_argument("--simulator", type=str, default="kinematic", choices=["kinematic", "dphysics", "none"])
#     p.add_argument("--dt", type=float, default=0.1)
#     p.add_argument("--control_v_scale", type=float, default=1.0)
#     p.add_argument("--control_w_scale", type=float, default=1.0)

#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--weight_decay", type=float, default=1e-4)
#     p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])

#     p.add_argument("--w_traj_pos", type=float, default=1.0)
#     p.add_argument("--w_traj_yaw", type=float, default=0.1)
#     p.add_argument("--w_terrain_sup", type=float, default=0.0)
#     p.add_argument("--w_height_cons", type=float, default=0.05)
#     p.add_argument("--w_tv", type=float, default=1e-4)
#     p.add_argument("--pose_loss", type=str, default="huber", choices=["huber", "smoothl1", "mse", "l1"])
#     p.add_argument("--huber_delta", type=float, default=1.0)

#     p.add_argument("--device", type=str, default="auto")
#     p.add_argument("--amp", action="store_true")
#     p.add_argument("--grad_clip", type=float, default=1.0)
#     p.add_argument("--log_dir", type=str, default="runs/f1_vault")
#     p.add_argument("--ckpt_dir", type=str, default="checkpoints/f1_vault")
#     p.add_argument("--save_every", type=int, default=1)

#     return p.parse_args()


# def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
#     out: Dict[str, Any] = {}
#     for k, v in batch.items():
#         if torch.is_tensor(v):
#             out[k] = v.to(device)
#         elif isinstance(v, dict):
#             out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
#         else:
#             out[k] = v
#     return out


# def main() -> None:
#     args = parse_args()
#     seed_everything(args.seed)

#     device = pick_device(args.device)
#     use_amp = bool(args.amp and device.type == "cuda")

#     if args.hdf5_path and Path(args.hdf5_path).exists():
#         from f1_vault.data.data import TerrainDataset, split_episodes, analyze_dataset_statistics
#         import h5py

#         stats = analyze_dataset_statistics(args.hdf5_path)
#         # print("Dataset stats:", json.dumps(stats, indent=2))
#         print("Dataset stats:", stats)

#         with h5py.File(args.hdf5_path, "r") as f:
#             all_eps = np.unique(f["episode_ids"][:])

#         train_eps, val_eps, _ = split_episodes(all_eps, train_ratio=0.7, val_ratio=0.15, random_seed=args.seed)

#         train_ds = TerrainDataset(args.hdf5_path, list(train_eps))
#         val_ds = TerrainDataset(args.hdf5_path, list(val_eps))

#         collate_fn = partial(
#             collate_chunks_with_scales,
#             control_v_scale=args.control_v_scale,
#             control_w_scale=args.control_w_scale,
#         )

#         train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
#         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
#     else:
#         train_ds = SyntheticTerrainTrajectoryDataset(n_samples=512, T=10, dt=args.dt)
#         val_ds = SyntheticTerrainTrajectoryDataset(n_samples=128, T=10, dt=args.dt)
#         train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_dict_batch)
#         val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_dict_batch)

#     model = TerrainEncoder(
#         input_channels=args.input_channels,
#         hidden_dims=list(args.hidden_dims),
#         min_stiffness=args.min_stiffness,
#         height_scale=args.height_scale,
#     ).to(device)

#     if args.weights:
#         sd = torch.load(args.weights, map_location="cpu")
#         if isinstance(sd, dict) and "model" in sd:
#             sd = sd["model"]
#         model.load_state_dict(sd, strict=False)

#     simulator: Optional[Callable[..., Any]] = None
#     if args.simulator == "kinematic":
#         simulator = KinematicUnicycleSimulator(dt=args.dt).to(device)
#     elif args.simulator == "dphysics":
#         simulator = try_make_dphysics_simulator(dt=args.dt)
#         if simulator is None:
#             simulator = KinematicUnicycleSimulator(dt=args.dt).to(device)

#     w = LossWeights(
#         traj_pos=args.w_traj_pos,
#         traj_yaw=args.w_traj_yaw,
#         terrain_supervised=args.w_terrain_sup,
#         height_consistency=args.w_height_cons,
#         tv=args.w_tv,
#     )
#     loss_fn = PhysicsInformedTerrainLoss(weights=w, pose_loss=args.pose_loss, huber_delta=args.huber_delta)

#     optimizer = build_optimizer(model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay)

#     scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

#     for epoch in range(args.epochs):
#         model.train()
#         for batch in train_loader:
#             batch = to_device(batch, device)
#             optimizer.zero_grad(set_to_none=True)

#             with torch.amp.autocast(device_type=device.type, enabled=use_amp):
#                 preds = model(batch["bev_map"])
#                 loss, logs = loss_fn(preds, batch, simulator=simulator)
#                 if not torch.isfinite(loss):
#                     print("💥 Non-finite loss. Logs:", logs)
#                     for k, v in batch.items():
#                         if torch.is_tensor(v):
#                             print(k, v.shape, "finite=", torch.isfinite(v).all().item(),
#                                 "min=", v.nan_to_num().min().item(), "max=", v.nan_to_num().max().item())
#                     break

#             if use_amp:
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss.backward()
#                 optimizer.step()

#         print(f"Epoch {epoch:03d} train_loss={logs.get('loss/total', float('nan')):.4f}")

#     # Checkpointing and TensorBoard hooks can be added similarly using torch.save and SummaryWriter.
#     # SummaryWriter is documented in torch.utils.tensorboard. citeturn2search0


# if __name__ == "__main__":
#     main()
"""
training.py

Training entrypoint for the F1-Vault TerrainEncoder.

Supports multiple simulators:
  - dphysics  : repo differentiable physics (if deps available)
  - kinematic : built-in differentiable unicycle model (always available)
  - none      : supervised/regularization only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from functools import partial

from f1_vault.models import TerrainEncoder
from f1_vault.losses import LossWeights, PhysicsInformedTerrainLoss


Tensor = torch.Tensor
def collate_chunks_with_scales(chunks, control_v_scale: float, control_w_scale: float):
    items = [c.to_tensors() for c in chunks]
    bev_map = torch.stack([it["bev_map"] for it in items], dim=0)
    initial_pose = torch.stack([it["initial_pose"] for it in items], dim=0)
    actions = torch.stack([it["actions"] for it in items], dim=0)
    gt_traj = torch.stack([it["gt_trajectory"] for it in items], dim=0)

    # IMPORTANT: keep raw dataset actions (throttle, steering) in [-1,1]
    # Let DPhysics convert internally via cfg.control_mode="throttle_steer"
    return {"bev_map": bev_map, "initial_pose": initial_pose, "actions": actions, "gt_trajectory": gt_traj}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quat_wxyz_from_yaw(yaw: Tensor) -> Tensor:
    half = 0.5 * yaw
    qw = torch.cos(half)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    qz = torch.sin(half)
    return torch.stack([qw, qx, qy, qz], dim=-1)


class KinematicUnicycleSimulator(nn.Module):
    """
    Differentiable fallback rollout.

    The dataset actions are collected as (throttle, steering), not raw (v, w).
    To match the dataset's next_state targets, this fallback converts throttle/steer
    to bicycle-model motion and returns POST-action states for each step.
    """

    def __init__(
        self,
        dt: float = 0.1,
        control_mode: str = "throttle_steer",
        throttle_to_v: float = 3.0,
        steer_to_delta: float = 0.488,
        wheelbase: float = 0.33,
        omega_clip: float = 6.0,
    ):
        super().__init__()
        self.dt = float(dt)
        self.control_mode = str(control_mode)
        self.throttle_to_v = float(throttle_to_v)
        self.steer_to_delta = float(steer_to_delta)
        self.wheelbase = float(wheelbase)
        self.omega_clip = float(omega_clip)

    def _controls_to_vw(self, controls: Tensor) -> tuple[Tensor, Tensor]:
        if self.control_mode == "throttle_steer":
            throttle = controls[..., 0]
            steer = controls[..., 1]
            v = throttle * self.throttle_to_v
            delta = steer * self.steer_to_delta
            w = v * torch.tan(delta) / max(self.wheelbase, 1e-6)
            if self.omega_clip > 0:
                w = w.clamp(-self.omega_clip, self.omega_clip)
            return v, w
        if self.control_mode == "vw":
            return controls[..., 0], controls[..., 1]
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")

    def forward(self, z_grid: Tensor, controls: Tensor, state=None, friction=None) -> Dict[str, Tensor]:
        B, T, _ = controls.shape
        v, w = self._controls_to_vw(controls)

        if state is not None and isinstance(state, (tuple, list)) and len(state) >= 3:
            x0 = state[0]  # (B,3)
            R0 = state[2]  # (B,3,3)
            x = x0[:, 0]
            y = x0[:, 1]
            yaw = torch.atan2(R0[:, 1, 0], R0[:, 0, 0] + 1e-8)
        else:
            x = torch.zeros(B, device=z_grid.device, dtype=z_grid.dtype)
            y = torch.zeros_like(x)
            yaw = torch.zeros_like(x)

        z0 = z_grid.mean(dim=(-2, -1))  # (B,)

        xs, ys, zs, yaws = [], [], [], []
        for t in range(T):
            x = x + v[:, t] * torch.cos(yaw) * self.dt
            y = y + v[:, t] * torch.sin(yaw) * self.dt
            yaw = yaw + w[:, t] * self.dt

            xs.append(x)
            ys.append(y)
            zs.append(z0)
            yaws.append(yaw)

        positions = torch.stack(
            [torch.stack(xs, dim=1), torch.stack(ys, dim=1), torch.stack(zs, dim=1)],
            dim=-1,
        )
        yaws = torch.stack(yaws, dim=1)
        return {"positions": positions, "yaws": yaws}


class SyntheticTerrainTrajectoryDataset(Dataset):
    """
    Minimal synthetic dataset for pipeline testing.
    """

    def __init__(self, n_samples: int = 512, H: int = 26, W: int = 26, T: int = 10, dt: float = 0.1):
        self.n_samples = int(n_samples)
        self.H, self.W, self.T = int(H), int(W), int(T)
        self.dt = float(dt)
        self.sim = KinematicUnicycleSimulator(dt=dt)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bev = torch.randn(1, self.H, self.W) * 0.1
        terrain_gt = {
            "height": bev[0],
            "stiffness": torch.ones(self.H, self.W) * 200,
            "damping": torch.ones(self.H, self.W) * 5,
            "friction": torch.ones(self.H, self.W) * 1.0,
        }
        controls = torch.randn(self.T, 2) * 0.2

        sim_out = self.sim(z_grid=terrain_gt["height"].unsqueeze(0), controls=controls.unsqueeze(0))
        pos = sim_out["positions"][0]  # (T,3)
        yaw = sim_out["yaws"][0]       # (T,)

        quat = quat_wxyz_from_yaw(yaw)
        gt_pose = torch.cat([pos, quat], dim=-1)

        return {
            "bev_map": bev,
            "initial_pose": torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
            "controls": controls,
            "gt_trajectory": gt_pose,
            "terrain_gt": terrain_gt,
        }


def collate_dict_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["bev_map"] = torch.stack([b["bev_map"] for b in batch], dim=0)
    out["initial_pose"] = torch.stack([b["initial_pose"] for b in batch], dim=0)
    out["controls"] = torch.stack([b["controls"] for b in batch], dim=0)
    out["gt_trajectory"] = torch.stack([b["gt_trajectory"] for b in batch], dim=0)

    if "terrain_gt" in batch[0]:
        keys = batch[0]["terrain_gt"].keys()
        out["terrain_gt"] = {k: torch.stack([b["terrain_gt"][k] for b in batch], dim=0) for k in keys}
    return out


def try_make_dphysics_simulator(dt: float) -> Optional[Callable[..., Any]]:
    """
    Attempt to build repo DPhysics simulator. Returns None if deps are missing.
    """
    try:
        from f1_vault.physics.physicscfg import DPhysConfig
        from f1_vault.physics.physics import DPhysics

        cfg = DPhysConfig()
        cfg.dt = float(dt)
        cfg.traj_sim_time = float(dt) * 10.0
        cfg.use_odeint = False

        # ensure throttle/steer mode is used
        cfg.control_mode = "throttle_steer"
        cfg.throttle_to_v = 3.0
        cfg.steer_to_delta = 0.488
        cfg.wheelbase = 0.33
        cfg.omega_clip = 6.0

        sim = DPhysics(dphys_cfg=cfg, device=str(pick_device("auto")))
        sim.eval()
        return sim
    except Exception as e:
        print(f"[WARN] DPhysics unavailable. Reason: {type(e).__name__}: {e}")
        return None


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train F1-Vault TerrainEncoder")

    p.add_argument("--hdf5_path", type=str, default="", help="Path to HDF5 dataset. If empty, use synthetic data.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--input_channels", type=int, default=1)
    p.add_argument("--hidden_dims", type=int, nargs="*", default=[32, 64, 64])
    p.add_argument("--min_stiffness", type=float, default=100.0)
    p.add_argument("--height_scale", type=float, default=0.5)
    p.add_argument("--weights", type=str, default="")

    p.add_argument("--simulator", type=str, default="kinematic", choices=["kinematic", "dphysics", "none"])
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--control_v_scale", type=float, default=1.0)
    p.add_argument("--control_w_scale", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])

    p.add_argument("--w_traj_pos", type=float, default=1.0)
    p.add_argument("--w_traj_yaw", type=float, default=0.1)
    p.add_argument("--w_terrain_sup", type=float, default=0.0)
    p.add_argument("--w_height_cons", type=float, default=0.05)
    p.add_argument("--w_tv", type=float, default=1e-4)
    p.add_argument("--pose_loss", type=str, default="huber", choices=["huber", "smoothl1", "mse", "l1"])
    p.add_argument("--huber_delta", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_dir", type=str, default="runs/f1_vault")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/f1_vault")
    p.add_argument("--save_every", type=int, default=1)

    return p.parse_args()


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    if args.hdf5_path and Path(args.hdf5_path).exists():
        from f1_vault.data.data import TerrainDataset, split_episodes, analyze_dataset_statistics
        import h5py

        stats = analyze_dataset_statistics(args.hdf5_path)
        # print("Dataset stats:", json.dumps(stats, indent=2))
        print("Dataset stats:", stats)

        with h5py.File(args.hdf5_path, "r") as f:
            all_eps = np.unique(f["episode_ids"][:])

        train_eps, val_eps, _ = split_episodes(all_eps, train_ratio=0.7, val_ratio=0.15, random_seed=args.seed)

        train_ds = TerrainDataset(args.hdf5_path, list(train_eps))
        val_ds = TerrainDataset(args.hdf5_path, list(val_eps))

        collate_fn = partial(
            collate_chunks_with_scales,
            control_v_scale=args.control_v_scale,
            control_w_scale=args.control_w_scale,
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        train_ds = SyntheticTerrainTrajectoryDataset(n_samples=512, T=10, dt=args.dt)
        val_ds = SyntheticTerrainTrajectoryDataset(n_samples=128, T=10, dt=args.dt)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_dict_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_dict_batch)

    model = TerrainEncoder(
        input_channels=args.input_channels,
        hidden_dims=list(args.hidden_dims),
        min_stiffness=args.min_stiffness,
        height_scale=args.height_scale,
    ).to(device)

    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)

    simulator: Optional[Callable[..., Any]] = None
    if args.simulator == "kinematic":
        simulator = KinematicUnicycleSimulator(dt=args.dt).to(device)
    elif args.simulator == "dphysics":
        simulator = try_make_dphysics_simulator(dt=args.dt)
        if simulator is None:
            simulator = KinematicUnicycleSimulator(dt=args.dt).to(device)

    w = LossWeights(
        traj_pos=args.w_traj_pos,
        traj_yaw=args.w_traj_yaw,
        terrain_supervised=args.w_terrain_sup,
        height_consistency=args.w_height_cons,
        tv=args.w_tv,
    )
    loss_fn = PhysicsInformedTerrainLoss(weights=w, pose_loss=args.pose_loss, huber_delta=args.huber_delta)

    optimizer = build_optimizer(model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(batch["bev_map"])
                loss, logs = loss_fn(preds, batch, simulator=simulator)
                if not torch.isfinite(loss):
                    print("💥 Non-finite loss. Logs:", logs)
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            print(k, v.shape, "finite=", torch.isfinite(v).all().item(),
                                "min=", v.nan_to_num().min().item(), "max=", v.nan_to_num().max().item())
                    break

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch:03d} train_loss={logs.get('loss/total', float('nan')):.4f}")

    # Checkpointing and TensorBoard hooks can be added similarly using torch.save and SummaryWriter.
    # SummaryWriter is documented in torch.utils.tensorboard. citeturn2search0


if __name__ == "__main__":
    main()