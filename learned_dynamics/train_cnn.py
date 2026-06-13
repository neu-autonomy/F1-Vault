"""
Train the one-step CNN dynamics model.

Hard-won lessons baked into this script (see CLAUDE.md for the full story):

  - Boundary filter is REQUIRED. Rows where `terminated` or `truncated` is
    True have `next_states[i]` set to the post-reset spawn position (a
    teleport, up to ~8m), not the result of stepping the simulator. Train
    AND eval must both filter them, or MAE numbers disagree by 10x.
  - Target-only normalization: inputs are fed RAW, only the prediction
    targets (deltas) are normalized. The original variance-weighting
    collapsed to uniform weights and let high-variance dims dominate.
  - Elevation map layout comes from the H5 attrs (currently 625 values =
    25x25, the LAST 625 dims of the state). Earlier versions hard-coded
    676 (26x26), which pulled 51 non-elevation state dims into the
    "elevation map" and scrambled its spatial structure. Checkpoints in
    models/cnn_dynamics* predate this fix.
  - Naive baselines (mean |true delta|, i.e. predict-zero) are printed at
    eval so a silently-broken model shows up immediately.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics_model import DynamicsCNN, grid_from_size


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    data_file = "data/raw/dynamics_data_0000.h5"
    # Elevation layout is read from the H5 attrs in main(); these are fallbacks.
    elevation_map_size = 625
    elevation_grid_size = 25
    random_seed = 42

    core_state_dim = 22

    elevation_latent_dim = 128
    hidden_dims = [256, 256]
    dropout = 0.1

    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    val_split = 0.2
    predict_delta = True

    use_normalization = True       # for downstream-tool compatibility
    use_loss_weighting = False
    handle_inf_elevation = True

    # v3 = first version trained on the correct 25x25 elevation layout.
    # (v2 and earlier checkpoints have a different decoder shape; don't mix.)
    save_dir = Path("models/cnn_dynamics_v3")
    checkpoint_every = 10

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")


# ============================================================================
# DATASET (target-only normalization)
# ============================================================================

class CNNDynamicsDataset(Dataset):
    def __init__(self, states, actions, next_states, config, target_stats=None):
        self.config = config

        if config.handle_inf_elevation:
            states = self._replace_inf(states.copy())
            next_states = self._replace_inf(next_states.copy())

        core_dim = config.core_state_dim
        elev_dim = config.elevation_map_size
        grid = config.elevation_grid_size

        self.core_states = states[:, :core_dim]
        self.elevation_maps = states[:, -elev_dim:].reshape(-1, grid, grid)
        self.next_core_states = next_states[:, :core_dim]
        self.next_elevation_maps = next_states[:, -elev_dim:].reshape(-1, grid, grid)
        self.actions = actions

        self.core_targets = self.next_core_states - self.core_states
        self.elevation_targets = self.next_elevation_maps - self.elevation_maps

        # Input stats (saved for downstream tools, not used in __getitem__)
        self.core_state_mean = self.core_states.mean(axis=0)
        self.core_state_std = self.core_states.std(axis=0) + 1e-8
        self.elevation_mean = self.elevation_maps.mean()
        self.elevation_std = self.elevation_maps.std() + 1e-8
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8

        # Target stats (used to normalize targets at __getitem__ time)
        if target_stats is None:
            self.core_target_mean = self.core_targets.mean(axis=0)
            self.core_target_std = self.core_targets.std(axis=0) + 1e-6
            self.elevation_target_mean = self.elevation_targets.mean()
            self.elevation_target_std = self.elevation_targets.std() + 1e-6
        else:
            self.core_target_mean = target_stats["core_target_mean"]
            self.core_target_std = target_stats["core_target_std"]
            self.elevation_target_mean = target_stats["elevation_target_mean"]
            self.elevation_target_std = target_stats["elevation_target_std"]

        # Compat fields for checkpoint format
        self.core_weights = np.ones(self.core_targets.shape[1])
        self.elevation_weight = 1.0

        print(f"\nDataset: {len(states):,} transitions")
        print(f"  Δcore mean[:3]: {self.core_target_mean[:3]}")
        print(f"  Δcore std[:3]:  {self.core_target_std[:3]}")
        print(f"  Δcore max abs[:3]: "
              f"{np.abs(self.core_targets[:, :3]).max(axis=0)}")

    def _replace_inf(self, data):
        elev = data[:, -self.config.elevation_map_size:]
        inf_mask = np.isinf(elev)
        if inf_mask.any():
            elev[inf_mask] = -10.0
            data[:, -self.config.elevation_map_size:] = elev
        return data

    def __len__(self):
        return len(self.core_states)

    def __getitem__(self, idx):
        core_state = self.core_states[idx].copy()
        elevation_map = self.elevation_maps[idx].copy()
        action = self.actions[idx].copy()

        core_target = (self.core_targets[idx]
                       - self.core_target_mean) / self.core_target_std
        elev_target = (self.elevation_targets[idx]
                       - self.elevation_target_mean) / self.elevation_target_std

        return (
            torch.FloatTensor(core_state),
            torch.FloatTensor(elevation_map[np.newaxis, :, :]),
            torch.FloatTensor(action),
            torch.FloatTensor(core_target),
            torch.FloatTensor(elev_target[np.newaxis, :, :]),
        )


# ============================================================================
# TRAIN / EVAL  (model lives in dynamics_model.py)
# ============================================================================

def loss_fn(pred_core, target_core, pred_elev, target_elev):
    core_loss = nn.functional.mse_loss(pred_core, target_core)
    elev_loss = nn.functional.mse_loss(pred_elev, target_elev)
    return core_loss + 0.1 * elev_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for core_states, elev_maps, actions, core_t, elev_t in loader:
        core_states = core_states.to(device)
        elev_maps = elev_maps.to(device)
        actions = actions.to(device)
        core_t = core_t.to(device)
        elev_t = elev_t.to(device)

        optimizer.zero_grad()
        pred_core, pred_elev = model(core_states, elev_maps, actions)
        loss = loss_fn(pred_core, core_t, pred_elev, elev_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(core_states)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, dataset):
    model.eval()
    total_loss = 0
    core_errors_phys, elev_errors_phys, true_deltas_phys = [], [], []

    core_t_mean = torch.from_numpy(dataset.core_target_mean).float().to(device)
    core_t_std = torch.from_numpy(dataset.core_target_std).float().to(device)
    elev_t_mean = float(dataset.elevation_target_mean)
    elev_t_std = float(dataset.elevation_target_std)

    with torch.no_grad():
        for core_states, elev_maps, actions, core_t, elev_t in loader:
            core_states = core_states.to(device)
            elev_maps = elev_maps.to(device)
            actions = actions.to(device)
            core_t = core_t.to(device)
            elev_t = elev_t.to(device)

            pred_core, pred_elev = model(core_states, elev_maps, actions)
            loss = loss_fn(pred_core, core_t, pred_elev, elev_t)
            total_loss += loss.item() * len(core_states)

            pred_core_phys = pred_core * core_t_std + core_t_mean
            target_core_phys = core_t * core_t_std + core_t_mean
            pred_elev_phys = pred_elev * elev_t_std + elev_t_mean
            target_elev_phys = elev_t * elev_t_std + elev_t_mean

            core_errors_phys.append((pred_core_phys - target_core_phys).cpu().numpy())
            elev_errors_phys.append((pred_elev_phys - target_elev_phys).cpu().numpy())
            true_deltas_phys.append(target_core_phys.cpu().numpy())

    return (total_loss / len(loader.dataset),
            np.concatenate(core_errors_phys),
            np.concatenate(elev_errors_phys),
            np.concatenate(true_deltas_phys))


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    config = Config()
    if args.data_file:  config.data_file = args.data_file
    if args.epochs:     config.num_epochs = args.epochs
    if args.batch_size: config.batch_size = args.batch_size
    if args.save_dir:   config.save_dir = Path(args.save_dir)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    print("=" * 80)
    print("CNN DYNAMICS TRAINING")
    print("=" * 80)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("\nLoading data...")
    with h5py.File(config.data_file, "r") as f:
        states = f["states"][:]
        actions = f["actions"][:]
        next_states = f["next_states"][:]
        terminated = f["terminated"][:]
        truncated = f["truncated"][:]
        action_dim = f.attrs["action_dim"]
        config.elevation_map_size = int(f.attrs.get("elevation_map_size",
                                                    config.elevation_map_size))
    config.elevation_grid_size = grid_from_size(config.elevation_map_size)

    print(f"  raw rows: {len(states):,}")
    print(f"  elevation map: last {config.elevation_map_size} dims "
          f"({config.elevation_grid_size}x{config.elevation_grid_size}, from H5 attrs)")

    # ---- Apply boundary filter (this is the key correction) ----
    valid = ~(terminated | truncated)
    n_total = len(states)
    n_kept = int(valid.sum())
    print(f"  boundary filter: keeping {n_kept:,} / {n_total:,} "
          f"({100*n_kept/n_total:.1f}%)")
    print(f"  (boundary rows have next_state = post-reset spawn, not "
          f"physics result)")

    states = states[valid]
    actions = actions[valid]
    next_states = next_states[valid]

    pos_delta = next_states[:, :3] - states[:, :3]
    print(f"  Δpos magnitudes (filtered): "
          f"|X| mean={np.abs(pos_delta[:,0]).mean():.3f}m  max={np.abs(pos_delta[:,0]).max():.3f}m")
    print(f"                              "
          f"|Y| mean={np.abs(pos_delta[:,1]).mean():.3f}m  max={np.abs(pos_delta[:,1]).max():.3f}m")
    print(f"                              "
          f"|Z| mean={np.abs(pos_delta[:,2]).mean():.3f}m  max={np.abs(pos_delta[:,2]).max():.3f}m")

    # ---- Train/val split ----
    n = len(states)
    n_train = int(n * (1 - config.val_split))
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    print(f"\nSplit: {len(train_idx):,} train, {len(val_idx):,} val")

    # ---- Datasets ----
    train_dataset = CNNDynamicsDataset(
        states[train_idx], actions[train_idx], next_states[train_idx], config,
    )
    val_dataset = CNNDynamicsDataset(
        states[val_idx], actions[val_idx], next_states[val_idx], config,
        target_stats={
            "core_target_mean": train_dataset.core_target_mean,
            "core_target_std":  train_dataset.core_target_std,
            "elevation_target_mean": train_dataset.elevation_target_mean,
            "elevation_target_std":  train_dataset.elevation_target_std,
        },
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=(config.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=(config.device == "cuda"),
    )

    # ---- Model ----
    print("\nInitializing model...")
    model = DynamicsCNN(
        core_state_dim=config.core_state_dim,
        action_dim=action_dim,
        elevation_latent_dim=config.elevation_latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        elevation_grid=config.elevation_grid_size,
    ).to(config.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # ---- Train ----
    print(f"\nTraining on {config.device}...")
    print("=" * 80)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device)
        val_loss, val_core_err, val_elev_err, val_true_delta = eval_epoch(
            model, val_loader, config.device, train_dataset,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Per-axis stats
        mae_per_axis = np.abs(val_core_err[:, :3]).mean(axis=0)
        naive_per_axis = np.abs(val_true_delta[:, :3]).mean(axis=0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")
            print(f"          Pos MAE  (m):    "
                  f"X={mae_per_axis[0]:.4f}  Y={mae_per_axis[1]:.4f}  Z={mae_per_axis[2]:.4f}")
            print(f"          Naive MAE(m):    "
                  f"X={naive_per_axis[0]:.4f}  Y={naive_per_axis[1]:.4f}  Z={naive_per_axis[2]:.4f}  "
                  f"(predict-zero baseline)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config_dict": {
                    "core_state_dim": config.core_state_dim,
                    "elevation_latent_dim": config.elevation_latent_dim,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "elevation_map_size": config.elevation_map_size,
                    "predict_delta": config.predict_delta,
                    "use_normalization": True,
                    "use_loss_weighting": False,
                    "target_only_normalization": True,
                    "trained_with_boundary_filter": True,
                },
                "stats": {
                    "core_state_mean": train_dataset.core_state_mean,
                    "core_state_std": train_dataset.core_state_std,
                    "elevation_mean": train_dataset.elevation_mean,
                    "elevation_std": train_dataset.elevation_std,
                    "action_mean": train_dataset.action_mean,
                    "action_std": train_dataset.action_std,
                    "core_target_mean": train_dataset.core_target_mean,
                    "core_target_std": train_dataset.core_target_std,
                    "elevation_target_mean": train_dataset.elevation_target_mean,
                    "elevation_target_std": train_dataset.elevation_target_std,
                    "core_weights": train_dataset.core_weights,
                    "elevation_weight": train_dataset.elevation_weight,
                },
            }, config.save_dir / "best_model.pt")

        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, config.save_dir / f"checkpoint_ep{epoch+1}.pt")

    # ---- Loss curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train", linewidth=2)
    ax.plot(val_losses, label="Val", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (normalized-MSE)")
    ax.set_title("CNN Training Progress")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.save_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ---- Final eval ----
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (best checkpoint, val set, boundary-filtered)")
    print("=" * 80)

    ckpt = torch.load(config.save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_core_err, _, final_true_delta = eval_epoch(
        model, val_loader, config.device, train_dataset,
    )

    print(f"\n  axis      model MAE      naive MAE       improvement")
    print(f"  " + "-" * 56)
    for i, lbl in enumerate(["X", "Y", "Z"]):
        m = np.abs(final_core_err[:, i]).mean()
        n_ = np.abs(final_true_delta[:, i]).mean()
        improvement = "BETTER" if m < n_ else "WORSE"
        ratio = m / n_ if n_ > 1e-9 else float("inf")
        print(f"  {lbl:>4}    {m*100:7.2f}cm     {n_*100:7.2f}cm     "
              f"{improvement} ({ratio:.2f}x naive)")

    print(f"\nIf model MAE >= naive MAE on any axis, the model is no better")
    print(f"than predicting 'no motion' on that axis. Either the model is")
    print(f"undertrained or that axis isn't predictable from the inputs.")
    print(f"\nModel saved: {config.save_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN dynamics model")
    parser.add_argument("--data_file", type=str, help="H5 data file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--save_dir", type=str, help="Output directory")
    args = parser.parse_args()
    main(args)