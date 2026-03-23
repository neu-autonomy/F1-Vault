# from __future__ import annotations

# import os
# import time
# import csv
# from pathlib import Path
# from typing import Any, Dict, Optional
# from functools import partial

# import numpy as np
# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader

# # Reuse your existing training module pieces (no edits to training.py)
# from f1_vault.training.training import (
#     parse_args,
#     seed_everything,
#     pick_device,
#     collate_chunks_with_scales,
#     to_device,
#     try_make_dphysics_simulator,
#     build_optimizer,
#     KinematicUnicycleSimulator,  # fallback sim class in training.py
# )

# from f1_vault.models import TerrainEncoder
# from f1_vault.losses import LossWeights, PhysicsInformedTerrainLoss


# def _make_loaders_from_hdf5(args) -> tuple[DataLoader, DataLoader]:
#     from f1_vault.data.data import TerrainDataset, split_episodes, analyze_dataset_statistics
#     import h5py

#     stats = analyze_dataset_statistics(args.hdf5_path)
#     print("Dataset stats:", stats)

#     with h5py.File(args.hdf5_path, "r") as f:
#         all_eps = np.unique(f["episode_ids"][:])

#     train_eps, val_eps, _ = split_episodes(
#         all_eps,
#         train_ratio=0.7,
#         val_ratio=0.15,
#         random_seed=args.seed,
#     )
#     train_ds = TerrainDataset(args.hdf5_path, list(train_eps))
#     val_ds = TerrainDataset(args.hdf5_path, list(val_eps))

#     collate_fn = partial(
#         collate_chunks_with_scales,
#         control_v_scale=args.control_v_scale,
#         control_w_scale=args.control_w_scale,
#     )

#     is_cuda = (pick_device(args.device).type == "cuda")
#     common = dict(
#         num_workers=args.num_workers,
#         collate_fn=collate_fn,
#     )

#     if is_cuda:
#         common.update(
#             dict(
#                 pin_memory=True,
#                 persistent_workers=(args.num_workers > 0),
#                 prefetch_factor=2 if args.num_workers > 0 else None,
#             )
#         )

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
#     return train_loader, val_loader


# def _make_simulator(args, device: torch.device):
#     sim = None
#     if args.simulator == "kinematic":
#         sim = KinematicUnicycleSimulator(dt=args.dt).to(device)
#     elif args.simulator == "dphysics":
#         sim = try_make_dphysics_simulator(dt=args.dt)
#         if sim is None:
#             sim = KinematicUnicycleSimulator(dt=args.dt).to(device)
#     elif args.simulator == "none":
#         sim = None
#     return sim


# def save_history_csv(history: list[Dict[str, float]], out_path: str) -> None:
#     if not history:
#         return

#     fieldnames = list(history[0].keys())
#     with open(out_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(history)


# def plot_history(history: list[Dict[str, float]], plot_dir: str) -> None:
#     if not history:
#         return

#     os.makedirs(plot_dir, exist_ok=True)

#     epochs = [int(row["epoch"]) for row in history]

#     def series(key: str):
#         vals = []
#         for row in history:
#             v = row.get(key, float("nan"))
#             vals.append(v)
#         return vals

#     train_loss = series("train_loss")
#     val_loss = series("val_loss")
#     val_pos_rmse = series("val_pos_rmse")
#     val_rmse_x = series("val_rmse_x")
#     val_rmse_y = series("val_rmse_y")
#     val_rmse_z = series("val_rmse_z")
#     val_rmse_xy = series("val_rmse_xy")

#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, train_loss, label="train_loss")
#     plt.plot(epochs, val_loss, label="val_loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Train / Val Loss")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.plot(epochs, val_pos_rmse, label="val_pos_rmse")
#     if not all(np.isnan(val_rmse_xy)):
#         plt.plot(epochs, val_rmse_xy, label="val_rmse_xy")
#     if not all(np.isnan(val_rmse_x)):
#         plt.plot(epochs, val_rmse_x, label="val_rmse_x")
#     if not all(np.isnan(val_rmse_y)):
#         plt.plot(epochs, val_rmse_y, label="val_rmse_y")
#     if not all(np.isnan(val_rmse_z)):
#         plt.plot(epochs, val_rmse_z, label="val_rmse_z")
#     plt.xlabel("Epoch")
#     plt.ylabel("RMSE")
#     plt.title("Validation RMSE")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, "rmse_curve.png"))
#     plt.close()


# def save_xy_trajectory_plot(pred_pos: np.ndarray, gt_pos: np.ndarray, out_path: str, title: str) -> None:
#     plt.figure(figsize=(6, 6))
#     plt.plot(gt_pos[:, 0], gt_pos[:, 1], label="gt_xy")
#     plt.plot(pred_pos[:, 0], pred_pos[:, 1], label="pred_xy")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title(title)
#     plt.grid(True)
#     plt.axis("equal")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()


# def save_z_plot(pred_pos: np.ndarray, gt_pos: np.ndarray, out_path: str, title: str) -> None:
#     plt.figure(figsize=(8, 5))
#     plt.plot(gt_pos[:, 2], label="gt_z")
#     plt.plot(pred_pos[:, 2], label="pred_z")
#     plt.xlabel("Step")
#     plt.ylabel("z")
#     plt.title(title)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()


# def _maybe_extract_pred_gt_positions(loss_fn, preds, batch, simulator):
#     """
#     Best-effort extraction of predicted and GT positions for per-axis RMSE and plots.
#     Returns:
#         pred_pos: [B, T, 3] or None
#         gt_pos:   [B, T, 3] or None
#     """
#     if simulator is None:
#         return None, None

#     try:
#         if "gt_trajectory" not in batch:
#             return None, None
#         if "initial_pose" not in batch:
#             return None, None

#         gt_pose = batch["gt_trajectory"]
#         gt_pos = gt_pose[..., 0:3]

#         if "actions" in batch:
#             controls = batch["actions"]
#         elif "controls" in batch:
#             controls = batch["controls"]
#         else:
#             return None, None

#         state0 = loss_fn._default_state0_from_pose(batch["initial_pose"])

#         sim_kwargs = {
#             "z_grid": preds["height"].clamp(-0.25, 0.25),
#             "controls": controls,
#             "state": state0,
#         }

#         if "friction" in preds:
#             sim_kwargs["friction"] = preds["friction"].clamp(0.2, 2.0)

#         sim_out = simulator(**sim_kwargs)
#         pred_pos, _ = loss_fn._parse_simulator_output(sim_out)

#         T = min(pred_pos.shape[1], gt_pos.shape[1])
#         pred_pos = pred_pos[:, :T]
#         gt_pos = gt_pos[:, :T]
#         return pred_pos, gt_pos

#     except Exception as e:
      
#         print(f"[DEBUG _maybe_extract_pred_gt_positions] {type(e).__name__}: {e}")
    
#         return None, None


# def evaluate(
#     model,
#     loss_fn,
#     val_loader,
#     device,
#     use_amp: bool,
#     simulator=None,
#     plot_examples: bool = False,
#     examples_dir: Optional[str] = None,
#     epoch: int = 0,
# ) -> Dict[str, float]:
#     model.eval()
#     losses = []
#     rmse = []
#     rmse_x = []
#     rmse_y = []
#     rmse_z = []
#     rmse_xy = []
#     example_saved = False

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(val_loader):
#             batch = to_device(batch, device)
#             if epoch == 0 and batch_idx == 0:
#                 print("BATCH KEYS:", list(batch.keys()))

#                 ctrl_key = None
#                 if "actions" in batch:
#                     ctrl_key = "actions"
#                 elif "controls" in batch:
#                     ctrl_key = "controls"

#                 if ctrl_key is not None:
#                     controls = batch[ctrl_key].detach().float().cpu()
#                     print(f"[DEBUG] control key: {ctrl_key}")
#                     print(f"[DEBUG] controls shape: {tuple(controls.shape)}")
#                     print(f"[DEBUG] controls overall min/max: {controls.min().item():.6f} / {controls.max().item():.6f}")
#                     print(f"[DEBUG] controls overall mean/std: {controls.mean().item():.6f} / {controls.std().item():.6f}")

#                     if controls.ndim >= 3 and controls.shape[-1] >= 2:
#                         v = controls[..., 0]
#                         w = controls[..., 1]
#                         print(f"[DEBUG] v min/max: {v.min().item():.6f} / {v.max().item():.6f}")
#                         print(f"[DEBUG] v mean/std: {v.mean().item():.6f} / {v.std().item():.6f}")
#                         print(f"[DEBUG] w min/max: {w.min().item():.6f} / {w.max().item():.6f}")
#                         print(f"[DEBUG] w mean/std: {w.mean().item():.6f} / {w.std().item():.6f}")

#                         print("[DEBUG] first sample controls[:5]:")
#                         print(controls[0, :5])
#             with torch.amp.autocast(device_type=device.type, enabled=use_amp):
#                 preds = model(batch["bev_map"])
#                 loss, logs = loss_fn(preds, batch, simulator=simulator)
#                 if isinstance(logs, dict):
#                     # if "metric/pos_rmse" in logs:
#                         # rmse.append(float(logs["metric/pos_rmse"]))
#                     if "metric/rmse_x" in logs:
#                         rmse_x.append(float(logs["metric/rmse_x"]))
#                     if "metric/rmse_y" in logs:
#                         rmse_y.append(float(logs["metric/rmse_y"]))
#                     if "metric/rmse_z" in logs:
#                         rmse_z.append(float(logs["metric/rmse_z"]))
#                     if "metric/rmse_xy" in logs:
#                         rmse_xy.append(float(logs["metric/rmse_xy"]))

#             losses.append(loss.detach().float().cpu())

#             if isinstance(logs, dict) and "metric/pos_rmse" in logs:
#                 try:
#                     rmse.append(float(logs["metric/pos_rmse"]))
#                 except Exception:
#                     pass

#             pred_pos, gt_pos = _maybe_extract_pred_gt_positions(loss_fn, preds, batch, simulator)
#             if pred_pos is not None and gt_pos is not None and plot_examples and plot_examples and (not example_saved) and examples_dir is not None:
#                 os.makedirs(examples_dir, exist_ok=True)

#                 pred_np = pred_pos[0].detach().float().cpu().numpy()
#                 gt_np = gt_pos[0].detach().float().cpu().numpy()

#                 save_xy_trajectory_plot(
#                     pred_np,
#                     gt_np,
#                     os.path.join(examples_dir, f"traj_epoch_{epoch:03d}.png"),
#                     f"Trajectory Overlay Epoch {epoch}",
#                 )
#                 save_z_plot(
#                     pred_np,
#                     gt_np,
#                     os.path.join(examples_dir, f"z_epoch_{epoch:03d}.png"),
#                     f"Z Plot Epoch {epoch}",
#                 )

#                 print(f"[DEBUG] saved trajectory plot -> {os.path.join(examples_dir, f'traj_epoch_{epoch:03d}.png')}")
#                 print(f"[DEBUG] saved z plot -> {os.path.join(examples_dir, f'z_epoch_{epoch:03d}.png')}")

#                 example_saved = True
#             if pred_pos is not None and gt_pos is not None:
#                 err = pred_pos - gt_pos
#                 err_xy = pred_pos[..., 0:2] - gt_pos[..., 0:2]

#                 rmse_x.append(torch.sqrt((err[..., 0] ** 2).mean()).item())
#                 rmse_y.append(torch.sqrt((err[..., 1] ** 2).mean()).item())
#                 rmse_z.append(torch.sqrt((err[..., 2] ** 2).mean()).item())
#                 rmse_xy.append(torch.sqrt((err_xy ** 2).mean()).item())

#                 if plot_examples and (not example_saved) and examples_dir is not None:
#                     os.makedirs(examples_dir, exist_ok=True)
#                     pred_np = pred_pos[0].detach().float().cpu().numpy()
#                     gt_np = gt_pos[0].detach().float().cpu().numpy()

#                     save_xy_trajectory_plot(
#                         pred_np,
#                         gt_np,
#                         os.path.join(examples_dir, f"traj_epoch_{epoch:03d}.png"),
#                         f"Trajectory Overlay Epoch {epoch}",
#                     )
#                     save_z_plot(
#                         pred_np,
#                         gt_np,
#                         os.path.join(examples_dir, f"z_epoch_{epoch:03d}.png"),
#                         f"Z Plot Epoch {epoch}",
#                     )
#                     example_saved = True

#     out = {
#         "val_loss": float(torch.stack(losses).mean().item()) if losses else float("inf"),
#         "val_pos_rmse": float(np.mean(rmse)) if rmse else float("nan"),
#         "val_rmse_x": float(np.mean(rmse_x)) if rmse_x else float("nan"),
#         "val_rmse_y": float(np.mean(rmse_y)) if rmse_y else float("nan"),
#         "val_rmse_z": float(np.mean(rmse_z)) if rmse_z else float("nan"),
#         "val_rmse_xy": float(np.mean(rmse_xy)) if rmse_xy else float("nan"),
#     }
#     return out


# def save_checkpoint(path: str, model, optimizer, scaler, epoch: int, metrics: Dict[str, float], args) -> None:
#     payload = {
#         "epoch": epoch,
#         "model": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "scaler": scaler.state_dict() if scaler is not None else None,
#         "metrics": metrics,
#         "args": vars(args),
#     }
#     torch.save(payload, path)


# def main():
#     args = parse_args()
#     if not hasattr(args, "accum_steps"):
#         args.accum_steps = 1
#     if not hasattr(args, "patience"):
#         args.patience = 10
#     if not hasattr(args, "plot_every"):
#         args.plot_every = 1

#     seed_everything(args.seed)
#     device = pick_device(args.device)
#     use_amp = bool(args.amp and device.type == "cuda")

#     if device.type == "cuda":
#         torch.backends.cudnn.benchmark = True
#         torch.set_float32_matmul_precision("high")

#     if args.hdf5_path and Path(args.hdf5_path).exists():
#         train_loader, val_loader = _make_loaders_from_hdf5(args)
#     else:
#         raise FileNotFoundError("Please pass a valid --hdf5_path to train_best.py")

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

#     simulator = _make_simulator(args, device)

#     w = LossWeights(
#         traj_pos=args.w_traj_pos,
#         traj_yaw=args.w_traj_yaw,
#         terrain_supervised=args.w_terrain_sup,
#         height_consistency=args.w_height_cons,
#         tv=args.w_tv,
#     )
#     loss_fn = PhysicsInformedTerrainLoss(
#         weights=w,
#         pose_loss=args.pose_loss,
#         huber_delta=args.huber_delta,
#     ).to(device)

#     optimizer = build_optimizer(
#         model,
#         args.optimizer,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )

#     scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

#     os.makedirs(args.ckpt_dir, exist_ok=True)
#     best_path = os.path.join(args.ckpt_dir, "best.pt")
#     last_path = os.path.join(args.ckpt_dir, "last.pt")

#     plot_dir = os.path.join(args.ckpt_dir, "plots")
#     examples_dir = os.path.join(plot_dir, "examples")
#     history_csv_path = os.path.join(args.ckpt_dir, "history.csv")

#     os.makedirs(plot_dir, exist_ok=True)
#     os.makedirs(examples_dir, exist_ok=True)

#     history: list[Dict[str, float]] = []

#     best_val = float("inf")
#     bad_epochs = 0
#     global_step = 0

#     for epoch in range(args.epochs):
#         model.train()
#         t0 = time.time()
#         running = []

#         optimizer.zero_grad(set_to_none=True)

#         for step, batch in enumerate(train_loader):
#             batch = to_device(batch, device)

#             with torch.amp.autocast(device_type=device.type, enabled=use_amp):
#                 preds = model(batch["bev_map"])
#                 loss, logs = loss_fn(preds, batch, simulator=simulator)

#                 if not torch.isfinite(loss):
#                     raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}. logs={logs}")

#                 loss_scaled = loss / max(int(args.accum_steps), 1)

#             if use_amp:
#                 assert scaler is not None
#                 scaler.scale(loss_scaled).backward()
#             else:
#                 loss_scaled.backward()

#             if (step + 1) % int(args.accum_steps) == 0:
#                 if args.grad_clip and args.grad_clip > 0:
#                     if use_amp:
#                         scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

#                 if use_amp:
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     optimizer.step()

#                 optimizer.zero_grad(set_to_none=True)

#             running.append(float(loss.detach().float().cpu().item()))
#             global_step += 1

#         # handle leftover gradients if len(train_loader) not divisible by accum_steps
#         if len(train_loader) % int(args.accum_steps) != 0:
#             if args.grad_clip and args.grad_clip > 0:
#                 if use_amp:
#                     scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

#             if use_amp:
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 optimizer.step()

#             optimizer.zero_grad(set_to_none=True)

#         train_loss = float(np.mean(running)) if running else float("nan")

#         metrics = evaluate(
#             model,
#             loss_fn,
#             val_loader,
#             device,
#             use_amp,
#             simulator=simulator,
#             plot_examples=(epoch % int(args.plot_every) == 0),
#             examples_dir=examples_dir,
#             epoch=epoch,
#         )

#         elapsed = time.time() - t0

#         row = {
#             "epoch": float(epoch),
#             "train_loss": float(train_loss),
#             "val_loss": float(metrics["val_loss"]),
#             "val_pos_rmse": float(metrics["val_pos_rmse"]),
#             "val_rmse_x": float(metrics["val_rmse_x"]),
#             "val_rmse_y": float(metrics["val_rmse_y"]),
#             "val_rmse_z": float(metrics["val_rmse_z"]),
#             "val_rmse_xy": float(metrics["val_rmse_xy"]),
#             "epoch_time_sec": float(elapsed),
#             "lr": float(optimizer.param_groups[0]["lr"]),
#         }
#         history.append(row)
#         save_history_csv(history, history_csv_path)
#         plot_history(history, plot_dir)

#         print(
#             f"Epoch {epoch:03d} | "
#             f"train_loss={train_loss:.6f} | "
#             f"val_loss={metrics['val_loss']:.6f} | "
#             f"val_pos_rmse={metrics['val_pos_rmse']:.6f} | "
#             f"val_rmse_xy={metrics['val_rmse_xy']:.6f} | "
#             f"val_rmse_z={metrics['val_rmse_z']:.6f} | "
#             f"time={elapsed:.1f}s"
#         )

#         save_checkpoint(last_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)

#         if metrics["val_loss"] < best_val:
#             best_val = metrics["val_loss"]
#             bad_epochs = 0
#             save_checkpoint(best_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)
#             print(f"✅ Saved BEST -> {best_path} (val_loss={best_val:.6f})")
#         else:
#             bad_epochs += 1

#         if bad_epochs >= int(args.patience):
#             print(f"⏹ Early stopping: no improvement for {args.patience} epochs.")
#             break

#     print("Done.")
#     print("Best checkpoint:", best_path)
#     print("Last checkpoint:", last_path)
#     print("History CSV:", history_csv_path)
#     print("Plots dir:", plot_dir)


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import os
import time
import csv
from pathlib import Path
from typing import Any, Dict, Optional
from functools import partial

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Reuse your existing training module pieces (no edits to training.py)
from f1_vault.training.training import (
    parse_args,
    seed_everything,
    pick_device,
    collate_chunks_with_scales,
    to_device,
    try_make_dphysics_simulator,
    build_optimizer,
    KinematicUnicycleSimulator,  # fallback sim class in training.py
)

from f1_vault.models import TerrainEncoder
from f1_vault.losses import LossWeights, PhysicsInformedTerrainLoss


def _make_loaders_from_hdf5(args) -> tuple[DataLoader, DataLoader]:
    from f1_vault.data.data import (
        TerrainDataset,
        split_episodes,
        analyze_dataset_statistics,
        make_trajectory_keys,
    )
    import h5py

    stats = analyze_dataset_statistics(args.hdf5_path)
    print("Dataset stats:", stats)

    with h5py.File(args.hdf5_path, "r") as f:
        env_ids = f["env_ids"][:] if "env_ids" in f else np.zeros_like(f["episode_ids"][:])
        episode_ids = f["episode_ids"][:]
        all_eps = np.unique(make_trajectory_keys(env_ids, episode_ids))

    train_eps, val_eps, _ = split_episodes(
        all_eps,
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=args.seed,
    )
    train_ds = TerrainDataset(args.hdf5_path, list(train_eps))
    val_ds = TerrainDataset(args.hdf5_path, list(val_eps))

    collate_fn = partial(
        collate_chunks_with_scales,
        control_v_scale=args.control_v_scale,
        control_w_scale=args.control_w_scale,
    )

    is_cuda = (pick_device(args.device).type == "cuda")
    common = dict(
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if is_cuda:
        common.update(
            dict(
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2 if args.num_workers > 0 else None,
            )
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def _make_simulator(args, device: torch.device):
    sim = None
    if args.simulator == "kinematic":
        sim = KinematicUnicycleSimulator(dt=args.dt).to(device)
    elif args.simulator == "dphysics":
        sim = try_make_dphysics_simulator(dt=args.dt)
        if sim is None:
            sim = KinematicUnicycleSimulator(dt=args.dt).to(device)
    elif args.simulator == "none":
        sim = None
    return sim


def save_history_csv(history: list[Dict[str, float]], out_path: str) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[Dict[str, float]], plot_dir: str) -> None:
    if not history:
        return

    os.makedirs(plot_dir, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]

    def series(key: str):
        vals = []
        for row in history:
            v = row.get(key, float("nan"))
            vals.append(v)
        return vals

    train_loss = series("train_loss")
    val_loss = series("val_loss")
    val_pos_rmse = series("val_pos_rmse")
    val_rmse_x = series("val_rmse_x")
    val_rmse_y = series("val_rmse_y")
    val_rmse_z = series("val_rmse_z")
    val_rmse_xy = series("val_rmse_xy")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_pos_rmse, label="val_pos_rmse")
    if not all(np.isnan(val_rmse_xy)):
        plt.plot(epochs, val_rmse_xy, label="val_rmse_xy")
    if not all(np.isnan(val_rmse_x)):
        plt.plot(epochs, val_rmse_x, label="val_rmse_x")
    if not all(np.isnan(val_rmse_y)):
        plt.plot(epochs, val_rmse_y, label="val_rmse_y")
    if not all(np.isnan(val_rmse_z)):
        plt.plot(epochs, val_rmse_z, label="val_rmse_z")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rmse_curve.png"))
    plt.close()


def save_xy_trajectory_plot(pred_pos: np.ndarray, gt_pos: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], label="gt_xy")
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], label="pred_xy")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_z_plot(pred_pos: np.ndarray, gt_pos: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(gt_pos[:, 2], label="gt_z")
    plt.plot(pred_pos[:, 2], label="pred_z")
    plt.xlabel("Step")
    plt.ylabel("z")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _maybe_extract_pred_gt_positions(loss_fn, preds, batch, simulator):
    """
    Best-effort extraction of predicted and GT positions for per-axis RMSE and plots.
    Returns:
        pred_pos: [B, T, 3] or None
        gt_pos:   [B, T, 3] or None
    """
    if simulator is None:
        return None, None

    try:
        if "gt_trajectory" not in batch:
            return None, None
        if "initial_pose" not in batch:
            return None, None

        gt_pose = batch["gt_trajectory"]
        gt_pos = gt_pose[..., 0:3]

        if "actions" in batch:
            controls = batch["actions"]
        elif "controls" in batch:
            controls = batch["controls"]
        else:
            return None, None

        state0 = loss_fn._default_state0_from_pose(batch["initial_pose"])

        sim_kwargs = {
            "z_grid": preds["height"].clamp(-0.25, 0.25),
            "controls": controls,
            "state": state0,
        }

        if "friction" in preds:
            sim_kwargs["friction"] = preds["friction"].clamp(0.2, 2.0)

        sim_out = simulator(**sim_kwargs)
        pred_pos, _ = loss_fn._parse_simulator_output(sim_out)

        T = min(pred_pos.shape[1], gt_pos.shape[1])
        pred_pos = pred_pos[:, :T]
        gt_pos = gt_pos[:, :T]
        return pred_pos, gt_pos

    except Exception as e:
      
        print(f"[DEBUG _maybe_extract_pred_gt_positions] {type(e).__name__}: {e}")
    
        return None, None


def evaluate(
    model,
    loss_fn,
    val_loader,
    device,
    use_amp: bool,
    simulator=None,
    plot_examples: bool = False,
    examples_dir: Optional[str] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    model.eval()
    losses = []
    rmse = []
    rmse_x = []
    rmse_y = []
    rmse_z = []
    rmse_xy = []
    example_saved = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = to_device(batch, device)
            if epoch == 0 and batch_idx == 0:
                print("BATCH KEYS:", list(batch.keys()))

                ctrl_key = None
                if "actions" in batch:
                    ctrl_key = "actions"
                elif "controls" in batch:
                    ctrl_key = "controls"

                if ctrl_key is not None:
                    controls = batch[ctrl_key].detach().float().cpu()
                    print(f"[DEBUG] control key: {ctrl_key}")
                    print(f"[DEBUG] controls shape: {tuple(controls.shape)}")
                    print(f"[DEBUG] controls overall min/max: {controls.min().item():.6f} / {controls.max().item():.6f}")
                    print(f"[DEBUG] controls overall mean/std: {controls.mean().item():.6f} / {controls.std().item():.6f}")

                    if controls.ndim >= 3 and controls.shape[-1] >= 2:
                        v = controls[..., 0]
                        w = controls[..., 1]
                        print(f"[DEBUG] v min/max: {v.min().item():.6f} / {v.max().item():.6f}")
                        print(f"[DEBUG] v mean/std: {v.mean().item():.6f} / {v.std().item():.6f}")
                        print(f"[DEBUG] w min/max: {w.min().item():.6f} / {w.max().item():.6f}")
                        print(f"[DEBUG] w mean/std: {w.mean().item():.6f} / {w.std().item():.6f}")

                        print("[DEBUG] first sample controls[:5]:")
                        print(controls[0, :5])
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(batch["bev_map"])
                loss, logs = loss_fn(preds, batch, simulator=simulator)
                if isinstance(logs, dict):
                    # if "metric/pos_rmse" in logs:
                        # rmse.append(float(logs["metric/pos_rmse"]))
                    if "metric/rmse_x" in logs:
                        rmse_x.append(float(logs["metric/rmse_x"]))
                    if "metric/rmse_y" in logs:
                        rmse_y.append(float(logs["metric/rmse_y"]))
                    if "metric/rmse_z" in logs:
                        rmse_z.append(float(logs["metric/rmse_z"]))
                    if "metric/rmse_xy" in logs:
                        rmse_xy.append(float(logs["metric/rmse_xy"]))

            losses.append(loss.detach().float().cpu())

            if isinstance(logs, dict) and "metric/pos_rmse" in logs:
                try:
                    rmse.append(float(logs["metric/pos_rmse"]))
                except Exception:
                    pass

            pred_pos, gt_pos = _maybe_extract_pred_gt_positions(loss_fn, preds, batch, simulator)
            if pred_pos is not None and gt_pos is not None and plot_examples and plot_examples and (not example_saved) and examples_dir is not None:
                os.makedirs(examples_dir, exist_ok=True)

                pred_np = pred_pos[0].detach().float().cpu().numpy()
                gt_np = gt_pos[0].detach().float().cpu().numpy()

                save_xy_trajectory_plot(
                    pred_np,
                    gt_np,
                    os.path.join(examples_dir, f"traj_epoch_{epoch:03d}.png"),
                    f"Trajectory Overlay Epoch {epoch}",
                )
                save_z_plot(
                    pred_np,
                    gt_np,
                    os.path.join(examples_dir, f"z_epoch_{epoch:03d}.png"),
                    f"Z Plot Epoch {epoch}",
                )

                print(f"[DEBUG] saved trajectory plot -> {os.path.join(examples_dir, f'traj_epoch_{epoch:03d}.png')}")
                print(f"[DEBUG] saved z plot -> {os.path.join(examples_dir, f'z_epoch_{epoch:03d}.png')}")

                example_saved = True
            if pred_pos is not None and gt_pos is not None:
                err = pred_pos - gt_pos
                err_xy = pred_pos[..., 0:2] - gt_pos[..., 0:2]

                rmse_x.append(torch.sqrt((err[..., 0] ** 2).mean()).item())
                rmse_y.append(torch.sqrt((err[..., 1] ** 2).mean()).item())
                rmse_z.append(torch.sqrt((err[..., 2] ** 2).mean()).item())
                rmse_xy.append(torch.sqrt((err_xy ** 2).mean()).item())

                if plot_examples and (not example_saved) and examples_dir is not None:
                    os.makedirs(examples_dir, exist_ok=True)
                    pred_np = pred_pos[0].detach().float().cpu().numpy()
                    gt_np = gt_pos[0].detach().float().cpu().numpy()

                    save_xy_trajectory_plot(
                        pred_np,
                        gt_np,
                        os.path.join(examples_dir, f"traj_epoch_{epoch:03d}.png"),
                        f"Trajectory Overlay Epoch {epoch}",
                    )
                    save_z_plot(
                        pred_np,
                        gt_np,
                        os.path.join(examples_dir, f"z_epoch_{epoch:03d}.png"),
                        f"Z Plot Epoch {epoch}",
                    )
                    example_saved = True

    out = {
        "val_loss": float(torch.stack(losses).mean().item()) if losses else float("inf"),
        "val_pos_rmse": float(np.mean(rmse)) if rmse else float("nan"),
        "val_rmse_x": float(np.mean(rmse_x)) if rmse_x else float("nan"),
        "val_rmse_y": float(np.mean(rmse_y)) if rmse_y else float("nan"),
        "val_rmse_z": float(np.mean(rmse_z)) if rmse_z else float("nan"),
        "val_rmse_xy": float(np.mean(rmse_xy)) if rmse_xy else float("nan"),
    }
    return out


def save_checkpoint(path: str, model, optimizer, scaler, epoch: int, metrics: Dict[str, float], args) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    if not hasattr(args, "accum_steps"):
        args.accum_steps = 1
    if not hasattr(args, "patience"):
        args.patience = 10
    if not hasattr(args, "plot_every"):
        args.plot_every = 1

    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if args.hdf5_path and Path(args.hdf5_path).exists():
        train_loader, val_loader = _make_loaders_from_hdf5(args)
    else:
        raise FileNotFoundError("Please pass a valid --hdf5_path to train_best.py")

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

    simulator = _make_simulator(args, device)

    w = LossWeights(
        traj_pos=args.w_traj_pos,
        traj_yaw=args.w_traj_yaw,
        terrain_supervised=args.w_terrain_sup,
        height_consistency=args.w_height_cons,
        tv=args.w_tv,
    )
    loss_fn = PhysicsInformedTerrainLoss(
        weights=w,
        pose_loss=args.pose_loss,
        huber_delta=args.huber_delta,
    ).to(device)

    optimizer = build_optimizer(
        model,
        args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")

    plot_dir = os.path.join(args.ckpt_dir, "plots")
    examples_dir = os.path.join(plot_dir, "examples")
    history_csv_path = os.path.join(args.ckpt_dir, "history.csv")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)

    history: list[Dict[str, float]] = []

    best_val = float("inf")
    bad_epochs = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = []

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(batch["bev_map"])
                loss, logs = loss_fn(preds, batch, simulator=simulator)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}. logs={logs}")

                loss_scaled = loss / max(int(args.accum_steps), 1)

            if use_amp:
                assert scaler is not None
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % int(args.accum_steps) == 0:
                if args.grad_clip and args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            running.append(float(loss.detach().float().cpu().item()))
            global_step += 1

        # handle leftover gradients if len(train_loader) not divisible by accum_steps
        if len(train_loader) % int(args.accum_steps) != 0:
            if args.grad_clip and args.grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(running)) if running else float("nan")

        metrics = evaluate(
            model,
            loss_fn,
            val_loader,
            device,
            use_amp,
            simulator=simulator,
            plot_examples=(epoch % int(args.plot_every) == 0),
            examples_dir=examples_dir,
            epoch=epoch,
        )

        elapsed = time.time() - t0

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(metrics["val_loss"]),
            "val_pos_rmse": float(metrics["val_pos_rmse"]),
            "val_rmse_x": float(metrics["val_rmse_x"]),
            "val_rmse_y": float(metrics["val_rmse_y"]),
            "val_rmse_z": float(metrics["val_rmse_z"]),
            "val_rmse_xy": float(metrics["val_rmse_xy"]),
            "epoch_time_sec": float(elapsed),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        save_history_csv(history, history_csv_path)
        plot_history(history, plot_dir)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"val_pos_rmse={metrics['val_pos_rmse']:.6f} | "
            f"val_rmse_xy={metrics['val_rmse_xy']:.6f} | "
            f"val_rmse_z={metrics['val_rmse_z']:.6f} | "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(last_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)

        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)
            print(f"✅ Saved BEST -> {best_path} (val_loss={best_val:.6f})")
        else:
            bad_epochs += 1

        if bad_epochs >= int(args.patience):
            print(f"⏹ Early stopping: no improvement for {args.patience} epochs.")
            break

    print("Done.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("History CSV:", history_csv_path)
    print("Plots dir:", plot_dir)


if __name__ == "__main__":
    main()