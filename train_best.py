# train_best.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from functools import partial

import numpy as np
import torch
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
    from f1_vault.data.data import TerrainDataset, split_episodes, analyze_dataset_statistics
    import h5py

    stats = analyze_dataset_statistics(args.hdf5_path)
    print("Dataset stats:", stats)

    with h5py.File(args.hdf5_path, "r") as f:
        all_eps = np.unique(f["episode_ids"][:])

    train_eps, val_eps, _ = split_episodes(all_eps, train_ratio=0.7, val_ratio=0.15, random_seed=args.seed)
    train_ds = TerrainDataset(args.hdf5_path, list(train_eps))
    val_ds = TerrainDataset(args.hdf5_path, list(val_eps))

    # Keep training.py behavior for controls scaling (since you asked to keep it as-is)
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

    # GPU throughput improvements
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
        else:
            # try_make_dphysics_simulator returns an eval()'d module already in your training.py :contentReference[oaicite:2]{index=2}
            pass
    elif args.simulator == "none":
        sim = None
    return sim


def evaluate(model, loss_fn, val_loader, device, use_amp: bool, simulator=None) -> Dict[str, float]:
    model.eval()
    losses = []
    rmse = []
    with torch.no_grad():
        for batch in val_loader:
            batch = to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(batch["bev_map"])
                loss, logs = loss_fn(preds, batch, simulator=simulator)

            losses.append(loss.detach().float().cpu())

            # loss.py populates metric/pos_rmse when sim+gt are present :contentReference[oaicite:3]{index=3}
            if isinstance(logs, dict) and "metric/pos_rmse" in logs:
                try:
                    rmse.append(float(logs["metric/pos_rmse"]))
                except Exception:
                    pass

    out = {"val_loss": float(torch.stack(losses).mean().item()) if losses else float("inf")}
    out["val_pos_rmse"] = float(np.mean(rmse)) if rmse else float("nan")
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
    # Extend args without touching training.py:
    # - accum_steps: gradient accumulation to simulate larger batch
    # - patience: early stopping
    args = parse_args()
    if not hasattr(args, "accum_steps"):
        args.accum_steps = 1
    if not hasattr(args, "patience"):
        args.patience = 10

    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    # GPU performance knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # enables TF32 on RTX for faster matmul

    # Data
    if args.hdf5_path and Path(args.hdf5_path).exists():
        train_loader, val_loader = _make_loaders_from_hdf5(args)
    else:
        raise FileNotFoundError("Please pass a valid --hdf5_path to train_best.py")

    # Model
    model = TerrainEncoder(
        input_channels=args.input_channels,
        hidden_dims=list(args.hidden_dims),
        min_stiffness=args.min_stiffness,
        height_scale=args.height_scale,
    ).to(device)

    # Optional preload weights (training.py supports either raw sd or {"model": sd} :contentReference[oaicite:4]{index=4})
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)

    simulator = _make_simulator(args, device)

    # Loss
    w = LossWeights(
        traj_pos=args.w_traj_pos,
        traj_yaw=args.w_traj_yaw,
        terrain_supervised=args.w_terrain_sup,
        height_consistency=args.w_height_cons,
        tv=args.w_tv,
    )
    loss_fn = PhysicsInformedTerrainLoss(weights=w, pose_loss=args.pose_loss, huber_delta=args.huber_delta).to(device)

    # Optimizer
    optimizer = build_optimizer(model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Checkpointing
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")

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
                # grad clip
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

        train_loss = float(np.mean(running)) if running else float("nan")
        metrics = evaluate(model, loss_fn, val_loader, device, use_amp, simulator=simulator)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={metrics['val_loss']:.6f} | val_pos_rmse={metrics['val_pos_rmse']:.6f} | "
            f"time={elapsed:.1f}s"
        )

        # Save last every epoch
        save_checkpoint(last_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)

        # Save best
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scaler, epoch, {"train_loss": train_loss, **metrics}, args)
            print(f"✅ Saved BEST -> {best_path} (val_loss={best_val:.6f})")
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= int(args.patience):
            print(f"⏹ Early stopping: no improvement for {args.patience} epochs.")
            break

    print("Done.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    main()