"""
CNN Dynamics Model Training

This script trains a CNN-based model that preserves spatial structure of elevation maps
instead of flattening them. The CNN encoder extracts spatial features from the heightmap,
which are then combined with core state and action to predict next state.

Key difference from MLP: Uses 2D convolutions on elevation map to capture spatial patterns
(hills, slopes, obstacles) instead of treating it as 676 independent values.

Author: [Your Name]
Date: 2026-01-31
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    data_file = 'data/raw/dynamics_data_0000.h5'
    elevation_map_size = 676  # 26x26 grid
    elevation_grid_size = 26   # Grid dimension
    
    # Model architecture
    elevation_latent_dim = 128  # CNN encoder output dimension
    hidden_dims = [256, 256]    # MLP hidden layers
    dropout = 0.1
    
    # Training
    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    val_split = 0.2
    
    # Prediction target
    predict_delta = True
    
    # Preprocessing
    normalize_states = False  # DISABLED - training on raw physical units
    normalize_actions = False  # DISABLED
    handle_inf_elevation = True
    
    # Output
    save_dir = Path('models/cnn_dynamics')
    checkpoint_every = 10
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


# ============================================================================
# DATASET (Separates elevation map for CNN)
# ============================================================================

class CNNDynamicsDataset(Dataset):
    """Dataset that separates elevation map for CNN processing."""
    
    def __init__(self, states, actions, next_states, config):
        """
        Args:
            states: (N, state_dim) current states
            actions: (N, action_dim) actions taken
            next_states: (N, state_dim) resulting next states
            config: Config object
        """
        self.config = config
        
        # Handle inf values
        if config.handle_inf_elevation:
            states = self._handle_inf(states.copy())
            next_states = self._handle_inf(next_states.copy())
        
        # Separate elevation map from core state
        self.elevation_maps = states[:, -config.elevation_map_size:].reshape(-1, config.elevation_grid_size, config.elevation_grid_size)
        self.core_states = states[:, :-config.elevation_map_size]
        
        self.next_elevation_maps = next_states[:, -config.elevation_map_size:].reshape(-1, config.elevation_grid_size, config.elevation_grid_size)
        self.next_core_states = next_states[:, :-config.elevation_map_size]
        
        self.actions = actions
        
        # Compute targets
        if config.predict_delta:
            self.core_targets = self.next_core_states - self.core_states
            self.elevation_targets = self.next_elevation_maps - self.elevation_maps
        else:
            self.core_targets = self.next_core_states
            self.elevation_targets = self.next_elevation_maps
        
        # Compute normalization statistics (for denormalization only - not used in training)
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
        
        print(f"Dataset size: {len(states):,} transitions")
        print(f"Core state dim: {self.core_states.shape[1]}")
        print(f"Elevation map: {config.elevation_grid_size}×{config.elevation_grid_size}")
        print(f"Action dim: {actions.shape[1]}")
        print(f"Predicting: {'state deltas' if config.predict_delta else 'next states'}")
        print(f"Normalization: DISABLED (training on raw physical units)")
    
    def _handle_inf(self, data):
        """Replace inf values with sentinel."""
        elevation_size = self.config.elevation_map_size
        elevation_maps = data[:, -elevation_size:]
        inf_mask = np.isinf(elevation_maps)
        if inf_mask.any():
            elevation_maps[inf_mask] = -10.0
            data[:, -elevation_size:] = elevation_maps
        return data
    
    def __len__(self):
        return len(self.core_states)
    
    def __getitem__(self, idx):
        core_state = self.core_states[idx]
        elevation_map = self.elevation_maps[idx]
        action = self.actions[idx]
        core_target = self.core_targets[idx]
        elevation_target = self.elevation_targets[idx]
        
        # NO NORMALIZATION - use raw physical values
        # (normalization disabled per mentor feedback)
        
        return (
            torch.FloatTensor(core_state),           # (core_state_dim,)
            torch.FloatTensor(elevation_map).unsqueeze(0),  # (1, 26, 26)
            torch.FloatTensor(action),               # (action_dim,)
            torch.FloatTensor(core_target),          # (core_state_dim,)
            torch.FloatTensor(elevation_target).unsqueeze(0)  # (1, 26, 26)
        )


# ============================================================================
# MODEL (CNN Encoder + MLP + CNN Decoder)
# ============================================================================

class ElevationEncoder(nn.Module):
    """CNN encoder for elevation map spatial features."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Input: (1, 26, 26)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (32, 13, 13)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (64, 6, 6)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → (128, 1, 1)
            
            nn.Flatten(),  # → (128,)
        )
        
        # Project to desired latent dimension
        self.projection = nn.Linear(128, latent_dim)
    
    def forward(self, elevation_map):
        """
        Args:
            elevation_map: (batch, 1, 26, 26)
        Returns:
            latent: (batch, latent_dim)
        """
        features = self.encoder(elevation_map)
        latent = self.projection(features)
        return latent


class ElevationDecoder(nn.Module):
    """CNN decoder to reconstruct elevation map from latent features."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Project latent to spatial features
        self.projection = nn.Linear(latent_dim, 128 * 7 * 7)
        
        self.decoder = nn.Sequential(
            # Reshape to (128, 7, 7)
            nn.Unflatten(1, (128, 7, 7)),
            
            # Upsample to 26×26
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # → (64, 14, 14)
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # → (32, 28, 28)
            nn.ReLU(),
            
            # Reduce to target size
            nn.Conv2d(32, 1, kernel_size=3, padding=0),  # → (1, 26, 26)
        )
    
    def forward(self, latent):
        """
        Args:
            latent: (batch, latent_dim)
        Returns:
            elevation_map: (batch, 1, 26, 26)
        """
        features = self.projection(latent)
        elevation_map = self.decoder(features)
        return elevation_map


class DynamicsCNN(nn.Module):
    """CNN-based dynamics model with spatial elevation encoding."""
    
    def __init__(self, core_state_dim, action_dim, elevation_latent_dim, hidden_dims, dropout=0.1):
        super().__init__()
        
        self.core_state_dim = core_state_dim
        self.action_dim = action_dim
        self.elevation_latent_dim = elevation_latent_dim
        
        # Elevation map encoder
        self.elevation_encoder = ElevationEncoder(latent_dim=elevation_latent_dim)
        
        # Dynamics predictor (MLP on combined features)
        input_dim = core_state_dim + action_dim + elevation_latent_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Predict both core state and elevation latent
        layers.append(nn.Linear(prev_dim, core_state_dim + elevation_latent_dim))
        
        self.dynamics = nn.Sequential(*layers)
        
        # Elevation map decoder
        self.elevation_decoder = ElevationDecoder(latent_dim=elevation_latent_dim)
        
        print(f"\nModel Architecture:")
        print(f"  Elevation Encoder: 26×26 → {elevation_latent_dim}-dim latent")
        print(f"  Dynamics MLP: {input_dim} → {hidden_dims} → {core_state_dim + elevation_latent_dim}")
        print(f"  Elevation Decoder: {elevation_latent_dim}-dim → 26×26")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def forward(self, core_state, elevation_map, action):
        """
        Args:
            core_state: (batch, core_state_dim)
            elevation_map: (batch, 1, 26, 26)
            action: (batch, action_dim)
        Returns:
            predicted_core_delta: (batch, core_state_dim)
            predicted_elevation_delta: (batch, 1, 26, 26)
        """
        batch_size = core_state.shape[0]
        
        # Encode elevation map to latent features
        elevation_latent = self.elevation_encoder(elevation_map)  # (batch, latent_dim)
        
        # Combine all inputs
        combined = torch.cat([core_state, action, elevation_latent], dim=-1)
        
        # Predict deltas
        prediction = self.dynamics(combined)  # (batch, core_state_dim + latent_dim)
        
        # Split prediction
        predicted_core = prediction[:, :self.core_state_dim]
        predicted_elevation_latent = prediction[:, self.core_state_dim:]
        
        # Decode elevation latent back to map
        predicted_elevation = self.elevation_decoder(predicted_elevation_latent)
        
        return predicted_core, predicted_elevation


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for core_states, elevation_maps, actions, core_targets, elevation_targets in loader:
        core_states = core_states.to(device)
        elevation_maps = elevation_maps.to(device)
        actions = actions.to(device)
        core_targets = core_targets.to(device)
        elevation_targets = elevation_targets.to(device)
        
        optimizer.zero_grad()
        
        pred_core, pred_elevation = model(core_states, elevation_maps, actions)
        
        # Combined loss (core state + elevation map)
        loss_core = criterion(pred_core, core_targets)
        loss_elevation = criterion(pred_elevation, elevation_targets)
        loss = loss_core + 0.1 * loss_elevation  # Weight elevation less (noisier)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(core_states)
    
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, dataset):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    core_errors = []
    elevation_errors = []
    
    with torch.no_grad():
        for core_states, elevation_maps, actions, core_targets, elevation_targets in loader:
            core_states = core_states.to(device)
            elevation_maps = elevation_maps.to(device)
            actions = actions.to(device)
            core_targets = core_targets.to(device)
            elevation_targets = elevation_targets.to(device)
            
            pred_core, pred_elevation = model(core_states, elevation_maps, actions)
            
            loss_core = criterion(pred_core, core_targets)
            loss_elevation = criterion(pred_elevation, elevation_targets)
            loss = loss_core + 0.1 * loss_elevation
            
            total_loss += loss.item() * len(core_states)
            
            # Errors are already in physical units (no normalization)
            pred_core_denorm = pred_core.cpu().numpy()
            target_core_denorm = core_targets.cpu().numpy()
            
            pred_elev_denorm = pred_elevation.cpu().numpy()
            target_elev_denorm = elevation_targets.cpu().numpy()
            
            core_errors.append(pred_core_denorm - target_core_denorm)
            elevation_errors.append(pred_elev_denorm - target_elev_denorm)
    
    avg_loss = total_loss / len(loader.dataset)
    core_errors = np.concatenate(core_errors, axis=0)
    elevation_errors = np.concatenate(elevation_errors, axis=0)
    
    return avg_loss, core_errors, elevation_errors


def compute_component_errors(core_errors, elevation_errors):
    """Compute errors for different components."""
    # First 3 components are XYZ position
    pos_errors = core_errors[:, :3]
    
    results = {
        'position': {
            'mae': np.abs(pos_errors).mean(axis=0),
            'rmse': np.sqrt((pos_errors**2).mean(axis=0))
        },
        'core_state': {
            'mae': np.abs(core_errors).mean(),
            'rmse': np.sqrt((core_errors**2).mean())
        },
        'elevation': {
            'mae': np.abs(elevation_errors).mean(),
            'rmse': np.sqrt((elevation_errors**2).mean())
        }
    }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    config = Config()
    
    # Override config
    if args.data_file:
        config.data_file = args.data_file
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print("=" * 80)
    print("CNN DYNAMICS MODEL TRAINING")
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
    
    print(f"Loaded {len(states):,} transitions")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    
    # Filter episode boundaries
    print("\n⚠️  Filtering out episode boundaries (resets)...")
    valid_mask = ~(terminated | truncated)
    
    print(f"  Before filtering: {len(states):,} transitions")
    print(f"  After filtering:  {valid_mask.sum():,} transitions ({100*valid_mask.sum()/len(states):.1f}%)")
    
    states = states[valid_mask]
    actions = actions[valid_mask]
    next_states = next_states[valid_mask]
    
    # Verify filtering
    pos_delta = next_states[:, :3] - states[:, :3]
    print(f"\n✓ Filtered position deltas:")
    print(f"  X: std={pos_delta[:,0].std():.4f}m, max={np.abs(pos_delta[:,0]).max():.4f}m")
    print(f"  Y: std={pos_delta[:,1].std():.4f}m, max={np.abs(pos_delta[:,1]).max():.4f}m")
    print(f"  Z: std={pos_delta[:,2].std():.4f}m, max={np.abs(pos_delta[:,2]).max():.4f}m")
    
    # Split data
    n_samples = len(states)
    n_train = int(n_samples * (1 - config.val_split))
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_idx):,} samples ({100*(1-config.val_split):.0f}%)")
    print(f"  Val:   {len(val_idx):,} samples ({100*config.val_split:.0f}%)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CNNDynamicsDataset(
        states[train_idx],
        actions[train_idx],
        next_states[train_idx],
        config
    )
    
    val_dataset = CNNDynamicsDataset(
        states[val_idx],
        actions[val_idx],
        next_states[val_idx],
        config
    )
    
    # Use same normalization
    val_dataset.core_state_mean = train_dataset.core_state_mean
    val_dataset.core_state_std = train_dataset.core_state_std
    val_dataset.elevation_mean = train_dataset.elevation_mean
    val_dataset.elevation_std = train_dataset.elevation_std
    val_dataset.action_mean = train_dataset.action_mean
    val_dataset.action_std = train_dataset.action_std
    val_dataset.core_target_mean = train_dataset.core_target_mean
    val_dataset.core_target_std = train_dataset.core_target_std
    val_dataset.elevation_target_mean = train_dataset.elevation_target_mean
    val_dataset.elevation_target_std = train_dataset.elevation_target_std
    
    # Create dataloaders
    use_pin_memory = config.device == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory
    )
    
    # Create model
    print("\nInitializing model...")
    core_state_dim = train_dataset.core_states.shape[1]
    
    model = DynamicsCNN(
        core_state_dim=core_state_dim,
        action_dim=action_dim,
        elevation_latent_dim=config.elevation_latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout
    ).to(config.device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Training loop
    print(f"\nTraining on {config.device}...")
    print("=" * 80)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        train_losses.append(train_loss)
        
        val_loss, val_core_errors, val_elev_errors = eval_epoch(model, val_loader, criterion, config.device, val_dataset)
        val_losses.append(val_loss)
        
        comp_errors = compute_component_errors(val_core_errors, val_elev_errors)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Pos MAE: [{comp_errors['position']['mae'][0]:.4f}, "
                  f"{comp_errors['position']['mae'][1]:.4f}, "
                  f"{comp_errors['position']['mae'][2]:.4f}]")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'elevation_latent_dim': config.elevation_latent_dim,
                    'hidden_dims': config.hidden_dims,
                    'dropout': config.dropout,
                    'elevation_map_size': config.elevation_map_size,
                    'predict_delta': config.predict_delta,
                },
                'normalization': {
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
                }
            }
            torch.save(checkpoint, config.save_dir / 'best_model.pt')
        
        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.save_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot training curves
    print("\nGenerating training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('CNN Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.save_dir / 'training_curves.png', dpi=150)
    print(f"Saved: {config.save_dir / 'training_curves.png'}")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    model.load_state_dict(torch.load(config.save_dir / 'best_model.pt', weights_only=False)['model_state_dict'])
    _, final_core_errors, final_elev_errors = eval_epoch(model, val_loader, criterion, config.device, val_dataset)
    final_comp_errors = compute_component_errors(final_core_errors, final_elev_errors)
    
    print(f"\nPosition Prediction Error:")
    print(f"  X: MAE={final_comp_errors['position']['mae'][0]:.4f}m, "
          f"RMSE={final_comp_errors['position']['rmse'][0]:.4f}m")
    print(f"  Y: MAE={final_comp_errors['position']['mae'][1]:.4f}m, "
          f"RMSE={final_comp_errors['position']['rmse'][1]:.4f}m")
    print(f"  Z: MAE={final_comp_errors['position']['mae'][2]:.4f}m, "
          f"RMSE={final_comp_errors['position']['rmse'][2]:.4f}m")
    
    print(f"\nCore State Error:")
    print(f"  MAE: {final_comp_errors['core_state']['mae']:.4f}")
    print(f"  RMSE: {final_comp_errors['core_state']['rmse']:.4f}")
    
    print(f"\nElevation Map Error:")
    print(f"  MAE: {final_comp_errors['elevation']['mae']:.4f}m")
    print(f"  RMSE: {final_comp_errors['elevation']['rmse']:.4f}m")
    
    print(f"\nModel saved to: {config.save_dir}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON WITH MLP BASELINE")
    print("=" * 80)
    print("\nMLP baseline: ~0.017m (1.7cm)")
    print(f"CNN result:   {final_comp_errors['position']['mae'].mean():.4f}m "
          f"({final_comp_errors['position']['mae'].mean()*100:.2f}cm)")
    
    improvement = (1 - final_comp_errors['position']['mae'].mean() / 0.017) * 100
    if improvement > 5:
        print(f"\n✓ CNN improved by {improvement:.1f}%! Spatial structure matters.")
    elif improvement < -5:
        print(f"\n⚠️  CNN performed {-improvement:.1f}% worse. Spatial encoding didn't help.")
    else:
        print(f"\n→ CNN similar to MLP ({improvement:+.1f}%). Spatial structure may not be critical.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN dynamics model")
    parser.add_argument('--data_file', type=str, help='Path to H5 data file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    args = parser.parse_args()
    
    main(args)