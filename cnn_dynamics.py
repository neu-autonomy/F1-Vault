"""
CNN Dynamics Model Training

Trains a CNN-based dynamics model that preserves spatial structure of elevation maps.
Supports both normalization and loss weighting approaches.

State vector layout (720 dims total):
    [0:3]     root_pos_w          (3)   -- position x, y, z (world)
    [3:7]     root_quat_w         (4)   -- orientation quaternion (w, x, y, z)
    [7:10]    world_euler_xyz     (3)   -- roll/pitch/yaw (redundant w/ quat, kept for simplicity)
    [10:13]   base_lin_vel        (3)   -- linear velocity (body frame)
    [13:16]   base_ang_vel        (3)   -- angular velocity (body frame)
    [16:19]   root_lin_vel_w      (3)   -- linear velocity (world frame)
    [19:22]   root_ang_vel_w      (3)   -- angular velocity (world frame)
    [22:44]   <joint pos/vel>     (22)  -- STRIPPED: not used by model
    [44:720]  elevation_map       (676) -- 26x26 BEV heightmap

Architecture:
- CNN Encoder: 26x26 elevation map -> 128-dim latent features
- MLP: core_state + action + elevation_latent -> predicted deltas
- CNN Decoder: elevation_latent -> 26x26 elevation map
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_file = 'data/raw/dynamics_data_0000.h5'
    elevation_map_size = 676  # 26x26 grid
    elevation_grid_size = 26
    random_seed = 42

    # State slicing (matches RobotState dataclass layout in the design doc)
    # Core state = root/base state only: pos(3) + quat(4) + euler(3) + base_lin(3)
    #              + base_ang(3) + root_lin_w(3) + root_ang_w(3) = 22 dims
    # Joints at [core_state_dim : -elevation_map_size] are intentionally stripped.
    core_state_dim = 22

    # Model
    elevation_latent_dim = 128
    hidden_dims = [256, 256]
    dropout = 0.1

    # Training
    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    val_split = 0.2
    predict_delta = True

    # Preprocessing (choose ONE approach)
    use_normalization = False      # Z-score normalization
    use_loss_weighting = True      # Inverse variance weighting
    handle_inf_elevation = True    # Replace inf with -10.0

    # Output
    save_dir = Path('models/cnn_dynamics')
    checkpoint_every = 10

    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


# ============================================================================
# DATASET
# ============================================================================

class CNNDynamicsDataset(Dataset):
    """Dataset with separate elevation map handling for CNN.

    Splits raw 720-dim state vectors into:
      - core_states:    first `core_state_dim` dims (root/base state, 22 dims)
      - elevation_maps: last `elevation_map_size` dims reshaped to 26x26

    The joint-position / joint-velocity block in the middle of the state
    (between index `core_state_dim` and the start of the elevation map) is
    dropped entirely -- it is not used as input, target, or supervision.
    """

    def __init__(self, states, actions, next_states, config):
        self.config = config

        # Handle inf values in the elevation slice before any slicing of state
        if config.handle_inf_elevation:
            states = self._replace_inf(states.copy())
            next_states = self._replace_inf(next_states.copy())

        # --- Split state: core (first 22) + elevation (last 676), drop middle ---
        core_dim = config.core_state_dim
        elev_dim = config.elevation_map_size
        grid = config.elevation_grid_size

        self.core_states = states[:, :core_dim]
        self.elevation_maps = states[:, -elev_dim:].reshape(-1, grid, grid)

        self.next_core_states = next_states[:, :core_dim]
        self.next_elevation_maps = next_states[:, -elev_dim:].reshape(-1, grid, grid)

        self.actions = actions

        # --- Targets (deltas or absolute) ---
        if config.predict_delta:
            self.core_targets = self.next_core_states - self.core_states
            self.elevation_targets = self.next_elevation_maps - self.elevation_maps
        else:
            self.core_targets = self.next_core_states
            self.elevation_targets = self.next_elevation_maps

        # --- Statistics (used by both normalization and weighting) ---
        self.core_state_mean = self.core_states.mean(axis=0)
        self.core_state_std = self.core_states.std(axis=0) + 1e-8
        self.elevation_mean = self.elevation_maps.mean()
        self.elevation_std = self.elevation_maps.std() + 1e-8
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8
        self.core_target_mean = self.core_targets.mean(axis=0)
        self.core_target_std = self.core_targets.std(axis=0) + 1e-8
        self.elevation_target_mean = self.elevation_targets.mean()
        self.elevation_target_std = self.elevation_targets.std() + 1e-8

        # --- Loss weights ---
        if config.use_loss_weighting:
            # Default: uniform weight on everything
            self.core_weights = np.ones(self.core_targets.shape[1])

            # Position (first 3 dims): weight by 1/var so XYZ error isn't drowned
            # out by larger-variance velocity/angular components.
            pos_var = self.core_targets[:, :3].var(axis=0).mean()
            self.core_weights[:3] = 1.0 / (pos_var + 1e-8)

            # Elevation: match position weighting magnitude
            elev_var = self.elevation_targets.var()
            self.elevation_weight = 1.0 / (elev_var + 1e-8)

            # Scale everything to [0, 1] to keep loss magnitudes sane
            max_weight = max(self.core_weights.max(), self.elevation_weight)
            self.core_weights = self.core_weights / max_weight
            self.elevation_weight = self.elevation_weight / max_weight

            print(f"\nLoss weighting:")
            print(f"  Position (X,Y,Z): [{self.core_weights[0]:.2f}, "
                  f"{self.core_weights[1]:.2f}, {self.core_weights[2]:.2f}]")
            print(f"  Core range: [{self.core_weights.min():.2f}, {self.core_weights.max():.2f}]")
            print(f"  Elevation: {self.elevation_weight:.4f}")
        else:
            self.core_weights = np.ones(self.core_targets.shape[1])
            self.elevation_weight = 1.0

        print(f"\nDataset: {len(states):,} transitions")
        print(f"  Raw state dim:  {states.shape[1]}")
        print(f"  Core state:     {self.core_states.shape[1]} dims  (kept: [0:{core_dim}])")
        print(f"  Joints dropped: {states.shape[1] - core_dim - elev_dim} dims  "
              f"(stripped: [{core_dim}:{states.shape[1] - elev_dim}])")
        print(f"  Elevation:      {grid}x{grid}  (kept: [-{elev_dim}:])")
        print(f"  Actions:        {actions.shape[1]} dims")
        print(f"  Target:         {'deltas' if config.predict_delta else 'absolute states'}")

        if config.use_normalization:
            print(f"  Preprocessing:  NORMALIZATION")
        elif config.use_loss_weighting:
            print(f"  Preprocessing:  LOSS WEIGHTING")
        else:
            print(f"  Preprocessing:  NONE")

    def _replace_inf(self, data):
        """Replace inf values in the elevation slice with a sentinel (-10.0)."""
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
        core_target = self.core_targets[idx].copy()
        elevation_target = self.elevation_targets[idx].copy()

        if self.config.use_normalization:
            core_state = (core_state - self.core_state_mean) / self.core_state_std
            elevation_map = (elevation_map - self.elevation_mean) / self.elevation_std
            action = (action - self.action_mean) / self.action_std
            core_target = (core_target - self.core_target_mean) / self.core_target_std
            elevation_target = (elevation_target - self.elevation_target_mean) / self.elevation_target_std

        return (
            torch.FloatTensor(core_state),
            torch.FloatTensor(elevation_map[np.newaxis, :, :]),  # add channel dim
            torch.FloatTensor(action),
            torch.FloatTensor(core_target),
            torch.FloatTensor(elevation_target[np.newaxis, :, :]),
        )


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class ElevationEncoder(nn.Module):
    """CNN encoder: 26x26 -> latent features."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 13x13

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 6x6

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),    # 1x1

            nn.Flatten(),
        )
        self.projection = nn.Linear(128, latent_dim)

    def forward(self, x):
        return self.projection(self.encoder(x))


class ElevationDecoder(nn.Module):
    """CNN decoder: latent -> 26x26."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.projection = nn.Linear(latent_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=0),                       # 26x26
        )

    def forward(self, x):
        return self.decoder(self.projection(x))


class DynamicsCNN(nn.Module):
    """Full dynamics model with CNN encoder/decoder.

    core_state_dim is whatever dimension the dataset yields (22 by default
    now that joint pos/vel are stripped). The MLP head size tracks it.
    """

    def __init__(self, core_state_dim, action_dim, elevation_latent_dim,
                 hidden_dims, dropout=0.1):
        super().__init__()

        self.core_state_dim = core_state_dim
        self.elevation_encoder = ElevationEncoder(elevation_latent_dim)

        input_dim = core_state_dim + action_dim + elevation_latent_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, core_state_dim + elevation_latent_dim))
        self.dynamics = nn.Sequential(*layers)

        self.elevation_decoder = ElevationDecoder(elevation_latent_dim)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nModel: {total_params:,} parameters")
        print(f"  Encoder: 26x26 -> {elevation_latent_dim}")
        print(f"  MLP:     {input_dim} -> {hidden_dims} -> {core_state_dim + elevation_latent_dim}")
        print(f"  Decoder: {elevation_latent_dim} -> 26x26")

    def forward(self, core_state, elevation_map, action):
        elev_latent = self.elevation_encoder(elevation_map)
        combined = torch.cat([core_state, action, elev_latent], dim=-1)
        prediction = self.dynamics(combined)

        pred_core = prediction[:, :self.core_state_dim]
        pred_elev_latent = prediction[:, self.core_state_dim:]
        pred_elevation = self.elevation_decoder(pred_elev_latent)

        return pred_core, pred_elevation


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_weighted_loss(pred_core, target_core, pred_elev, target_elev,
                          core_weights, elev_weight, device):
    """Weighted MSE loss."""
    w_core = torch.FloatTensor(core_weights).to(device)

    core_sq_err = (pred_core - target_core) ** 2
    elev_sq_err = (pred_elev - target_elev) ** 2

    weighted_core = (core_sq_err * w_core).mean()
    weighted_elev = elev_sq_err.mean() * elev_weight

    return weighted_core + weighted_elev


def compute_standard_loss(pred_core, target_core, pred_elev, target_elev):
    """Standard MSE loss with fixed elevation weighting."""
    core_loss = nn.functional.mse_loss(pred_core, target_core)
    elev_loss = nn.functional.mse_loss(pred_elev, target_elev)
    return core_loss + 0.1 * elev_loss


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device, dataset, config):
    model.train()
    total_loss = 0

    for core_states, elev_maps, actions, core_targets, elev_targets in loader:
        core_states = core_states.to(device)
        elev_maps = elev_maps.to(device)
        actions = actions.to(device)
        core_targets = core_targets.to(device)
        elev_targets = elev_targets.to(device)

        optimizer.zero_grad()
        pred_core, pred_elev = model(core_states, elev_maps, actions)

        if config.use_loss_weighting:
            loss = compute_weighted_loss(
                pred_core, core_targets, pred_elev, elev_targets,
                dataset.core_weights, dataset.elevation_weight, device,
            )
        else:
            loss = compute_standard_loss(pred_core, core_targets, pred_elev, elev_targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(core_states)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, dataset, config):
    model.eval()
    total_loss = 0
    core_errors = []
    elev_errors = []

    with torch.no_grad():
        for core_states, elev_maps, actions, core_targets, elev_targets in loader:
            core_states = core_states.to(device)
            elev_maps = elev_maps.to(device)
            actions = actions.to(device)
            core_targets = core_targets.to(device)
            elev_targets = elev_targets.to(device)

            pred_core, pred_elev = model(core_states, elev_maps, actions)

            if config.use_loss_weighting:
                loss = compute_weighted_loss(
                    pred_core, core_targets, pred_elev, elev_targets,
                    dataset.core_weights, dataset.elevation_weight, device,
                )
            else:
                loss = compute_standard_loss(pred_core, core_targets, pred_elev, elev_targets)

            total_loss += loss.item() * len(core_states)

            # Physical-unit errors for reporting
            if config.use_normalization:
                pred_core_phys = pred_core.cpu().numpy() * dataset.core_target_std + dataset.core_target_mean
                target_core_phys = core_targets.cpu().numpy() * dataset.core_target_std + dataset.core_target_mean
                pred_elev_phys = pred_elev.cpu().numpy() * dataset.elevation_target_std + dataset.elevation_target_mean
                target_elev_phys = elev_targets.cpu().numpy() * dataset.elevation_target_std + dataset.elevation_target_mean
            else:
                pred_core_phys = pred_core.cpu().numpy()
                target_core_phys = core_targets.cpu().numpy()
                pred_elev_phys = pred_elev.cpu().numpy()
                target_elev_phys = elev_targets.cpu().numpy()

            core_errors.append(pred_core_phys - target_core_phys)
            elev_errors.append(pred_elev_phys - target_elev_phys)

    return total_loss / len(loader.dataset), np.concatenate(core_errors), np.concatenate(elev_errors)


def compute_metrics(core_errors, elev_errors):
    pos_errors = core_errors[:, :3]  # XYZ

    return {
        'position': {
            'mae': np.abs(pos_errors).mean(axis=0),
            'rmse': np.sqrt((pos_errors ** 2).mean(axis=0)),
        },
        'core_state': {
            'mae': np.abs(core_errors).mean(),
            'rmse': np.sqrt((core_errors ** 2).mean()),
        },
        'elevation': {
            'mae': np.abs(elev_errors).mean(),
            'rmse': np.sqrt((elev_errors ** 2).mean()),
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    config = Config()

    if args.data_file:
        config.data_file = args.data_file
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.mode:
        config.use_normalization = (args.mode == 'normalize')
        config.use_loss_weighting = (args.mode == 'weight')

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    print("=" * 80)
    print("CNN DYNAMICS MODEL (joints stripped)")
    print("=" * 80)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    with h5py.File(config.data_file, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]
        next_states = f['next_states'][:]
        terminated = f['terminated'][:]
        truncated = f['truncated'][:]
        state_dim = f.attrs['state_dim']
        action_dim = f.attrs['action_dim']

    print(f"  {len(states):,} transitions")
    print(f"  State: {state_dim} dims, Action: {action_dim} dims")

    # Sanity-check layout: raw state should be core + joints + elevation
    expected_min = config.core_state_dim + config.elevation_map_size
    if states.shape[1] < expected_min:
        raise ValueError(
            f"Raw state dim {states.shape[1]} is smaller than core ({config.core_state_dim}) "
            f"+ elevation ({config.elevation_map_size}). Check core_state_dim / elevation_map_size."
        )

    # Filter episode boundaries
    print("\nFiltering episode boundaries...")
    valid = ~(terminated | truncated)
    print(f"  Kept: {valid.sum():,} / {len(states):,} ({100 * valid.sum() / len(states):.1f}%)")

    states = states[valid]
    actions = actions[valid]
    next_states = next_states[valid]

    pos_delta = next_states[:, :3] - states[:, :3]
    print(f"  Max delta: X={np.abs(pos_delta[:,0]).max():.3f}m, "
          f"Y={np.abs(pos_delta[:,1]).max():.3f}m, Z={np.abs(pos_delta[:,2]).max():.3f}m")

    # Split
    n = len(states)
    n_train = int(n * (1 - config.val_split))
    idx = np.random.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    print(f"\nSplit: {len(train_idx):,} train, {len(val_idx):,} val")

    # Datasets
    print("\nCreating datasets...")
    train_dataset = CNNDynamicsDataset(
        states[train_idx], actions[train_idx], next_states[train_idx], config,
    )
    val_dataset = CNNDynamicsDataset(
        states[val_idx], actions[val_idx], next_states[val_idx], config,
    )

    # Share normalization stats + weights between splits
    for attr in (
        'core_state_mean', 'core_state_std',
        'elevation_mean', 'elevation_std',
        'action_mean', 'action_std',
        'core_target_mean', 'core_target_std',
        'elevation_target_mean', 'elevation_target_std',
        'core_weights', 'elevation_weight',
    ):
        setattr(val_dataset, attr, getattr(train_dataset, attr))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=(config.device == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=(config.device == 'cuda'),
    )

    # Model
    print("\nInitializing model...")
    model = DynamicsCNN(
        core_state_dim=train_dataset.core_states.shape[1],  # = 22 by default
        action_dim=action_dim,
        elevation_latent_dim=config.elevation_latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)

    # Train
    print(f"\nTraining on {config.device}...")
    print("=" * 80)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config.device,
                                 train_dataset, config)
        val_loss, val_core_err, val_elev_err = eval_epoch(
            model, val_loader, config.device, val_dataset, config,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics = compute_metrics(val_core_err, val_elev_err)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Pos MAE: [{metrics['position']['mae'][0]:.4f}, "
                  f"{metrics['position']['mae'][1]:.4f}, "
                  f"{metrics['position']['mae'][2]:.4f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config_dict': {
                    'core_state_dim': config.core_state_dim,
                    'elevation_latent_dim': config.elevation_latent_dim,
                    'hidden_dims': config.hidden_dims,
                    'dropout': config.dropout,
                    'elevation_map_size': config.elevation_map_size,
                    'predict_delta': config.predict_delta,
                    'use_normalization': config.use_normalization,
                    'use_loss_weighting': config.use_loss_weighting,
                },
                'stats': {
                    'core_state_mean': train_dataset.core_state_mean,
                    'core_state_std': train_dataset.core_state_std,
                    'elevation_mean': train_dataset.elevation_mean,
                    'elevation_std': train_dataset.elevation_std,
                    'action_mean': train_dataset.action_mean,
                    'action_std': train_dataset.action_std,
                    'core_target_mean': train_dataset.core_target_mean,
                    'core_target_std': train_dataset.core_target_std,
                    'elevation_target_mean': train_dataset.elevation_target_mean,
                    'elevation_target_std': train_dataset.elevation_target_std,
                    'core_weights': train_dataset.core_weights,
                    'elevation_weight': train_dataset.elevation_weight,
                },
            }, config.save_dir / 'best_model.pt')

        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.save_dir / f'checkpoint_ep{epoch+1}.pt')

    print("\n" + "=" * 80)
    print(f"Training complete! Best val loss: {best_val_loss:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('CNN Training Progress (joints stripped)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.save_dir / 'training_curves.png', dpi=150)
    print(f"Saved: {config.save_dir / 'training_curves.png'}")

    # Final eval
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    checkpoint = torch.load(config.save_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_core_err, final_elev_err = eval_epoch(
        model, val_loader, config.device, val_dataset, config,
    )
    final_metrics = compute_metrics(final_core_err, final_elev_err)

    print(f"\nPosition Error:")
    print(f"  X: MAE={final_metrics['position']['mae'][0]:.4f}m, "
          f"RMSE={final_metrics['position']['rmse'][0]:.4f}m")
    print(f"  Y: MAE={final_metrics['position']['mae'][1]:.4f}m, "
          f"RMSE={final_metrics['position']['rmse'][1]:.4f}m")
    print(f"  Z: MAE={final_metrics['position']['mae'][2]:.4f}m, "
          f"RMSE={final_metrics['position']['rmse'][2]:.4f}m")

    print(f"\nCore State: MAE={final_metrics['core_state']['mae']:.4f}, "
          f"RMSE={final_metrics['core_state']['rmse']:.4f}")
    print(f"Elevation:  MAE={final_metrics['elevation']['mae']:.4f}m, "
          f"RMSE={final_metrics['elevation']['rmse']:.4f}m")

    avg_pos_mae = final_metrics['position']['mae'].mean()
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"MLP baseline: 0.017m (1.7cm)")
    print(f"CNN result:   {avg_pos_mae:.4f}m ({avg_pos_mae*100:.2f}cm)")

    improvement = (1 - avg_pos_mae / 0.017) * 100
    if improvement > 5:
        print(f"CNN improved by {improvement:.1f}%")
    elif improvement < -5:
        print(f"CNN worse by {-improvement:.1f}%")
    else:
        print(f"Similar performance ({improvement:+.1f}%)")

    print(f"\nModel saved: {config.save_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN dynamics model (joints stripped)')
    parser.add_argument('--data_file', type=str, help='H5 data file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--mode', type=str, choices=['normalize', 'weight', 'none'],
                        help='Preprocessing: normalize, weight, or none')
    args = parser.parse_args()

    main(args)