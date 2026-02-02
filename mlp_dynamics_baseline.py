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

"""
Config class to specify data/model sizes, devices, and files.
"""
class Config:
    # Data
    data_file = 'data/raw/dynamics_data_0000.h5'
    elevation_map_size = 676 # 26 x 26
    
    # Model
    hidden_dims = [512, 512, 512]
    activation = 'relu'
    dropout = 0.1
    
    # Training
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    val_split = 0.2
    
    # Prediction target
    predict_delta = True  # predict (s_{t+1} - s_t) instead of s_{t+1}
    
    # Preprocessing
    normalize_states = True
    normalize_actions = True
    handle_inf_elevation = True  # inf --> sentinel value
    
    # Output
    save_dir = Path('models/baseline_dynamics')
    checkpoint_every = 10
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


"""
Dataset processing for dynamics model training.
"""
class DynamicsDataset(Dataset):    
    """
    Args:
        states: (N, state_dim) current states
        actions: (N, action_dim) actions taken
        next_states: (N, state_dim) resulting next states
        config: Config object
    """
    def __init__(self, states, actions, next_states, config):
        self.config = config
        
        # Handle inf values in elevation maps
        if config.handle_inf_elevation:
            states = self._handle_inf(states.copy())
            next_states = self._handle_inf(next_states.copy())
        
        # Compute state deltas
        if config.predict_delta:
            self.targets = next_states - states  # Predict change
        else:
            self.targets = next_states  # Predict absolute next state
        
        self.states = states
        self.actions = actions
        
        # Compute normalization statistics
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-8
        
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8
        
        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0) + 1e-8
        
        # print(f"Dataset size: {len(states):,} transitions")
        # print(f"State dim: {states.shape[1]}")
        # print(f"Action dim: {actions.shape[1]}")
        # print(f"Predicting: {'state deltas' if config.predict_delta else 'next states'}")
    
    """
    Replace inf values with a large negative sentinel.
    """
    def _handle_inf(self, data):
        elevation_size = self.config.elevation_map_size
        elevation_maps = data[:, -elevation_size:]
        
        inf_mask = np.isinf(elevation_maps)
        if inf_mask.any():
            print(f"  Replacing {inf_mask.sum():,} inf values in elevation maps")
            elevation_maps[inf_mask] = -10.0
            data[:, -elevation_size:] = elevation_maps
        
        return data
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        target = self.targets[idx]
        
        # Normalize
        if self.config.normalize_states:
            state = (state - self.state_mean) / self.state_std
            target = (target - self.target_mean) / self.target_std
        
        if self.config.normalize_actions:
            action = (action - self.action_mean) / self.action_std
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(target)
        )


"""
MLP model for dynamics prediction.
"""
class DynamicsMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, dropout=0.1):
        super().__init__()
        
        input_dim = state_dim + action_dim
        output_dim = state_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # print(f"\nModel Architecture:")
        # print(f"  Input: {input_dim} (state + action)")
        # print(f"  Hidden layers: {hidden_dims}")
        # print(f"  Output: {output_dim} (state)")
        # total_params = sum(p.numel() for p in self.parameters())
        # print(f"  Total parameters: {total_params:,}")
    
    """
    Args:
        state: (batch, state_dim)
        action: (batch, action_dim)
    Returns:
        predicted_next_state or predicted_delta: (batch, state_dim)
    """
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


"""
MLP Training.
"""
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for states, actions, targets in loader:
        states = states.to(device)
        actions = actions.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(states, actions)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(states)
    
    return total_loss / len(loader.dataset)

"""
Evaluate on validation set.
"""
def eval_epoch(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    
    errors = []
    
    with torch.no_grad():
        for states, actions, targets in loader:
            states = states.to(device)
            actions = actions.to(device)
            targets = targets.to(device)
            
            predictions = model(states, actions)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(states)
            
            pred_denorm = predictions.cpu().numpy() * dataset.target_std + dataset.target_mean
            target_denorm = targets.cpu().numpy() * dataset.target_std + dataset.target_mean
            
            errors.append(pred_denorm - target_denorm)
    
    avg_loss = total_loss / len(loader.dataset)
    errors = np.concatenate(errors, axis=0)
    
    return avg_loss, errors


def compute_component_errors(errors, state_dim, elevation_map_size):
    core_dim = state_dim - elevation_map_size
    core_errors = errors[:, :core_dim]
    elevation_errors = errors[:, core_dim:]
    
    pos_errors = errors[:, :3]  # XYZ position
    
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


"""
Main client to run MLP.
"""
def main(args):
    config = Config()
    
    if args.data_file:
        config.data_file = args.data_file
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print("MLP Baseline Dynamics Model Training")
    print("=" * 80)
    
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Filter out episode boundary transitions
    valid_mask = ~(terminated | truncated)
    
    # Only keep valid transitions
    states = states[valid_mask]
    actions = actions[valid_mask]
    next_states = next_states[valid_mask]
    
    # Verify filtering worked
    pos_delta = next_states[:, :3] - states[:, :3]
    
    if np.abs(pos_delta).max() > 5.0:
        print("\nWarning: there might be issues with transition data.")
    
    # Split data
    n_samples = len(states)
    n_train = int(n_samples * (1 - config.val_split))
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DynamicsDataset(
        states[train_idx],
        actions[train_idx],
        next_states[train_idx],
        config
    )
    
    val_dataset = DynamicsDataset(
        states[val_idx],
        actions[val_idx],
        next_states[val_idx],
        config
    )
    
    # Use same normalization as training set
    val_dataset.state_mean = train_dataset.state_mean
    val_dataset.state_std = train_dataset.state_std
    val_dataset.action_mean = train_dataset.action_mean
    val_dataset.action_std = train_dataset.action_std
    val_dataset.target_mean = train_dataset.target_mean
    val_dataset.target_std = train_dataset.target_std
    
    # Create dataloaders
    use_pin_memory = config.device == 'cuda'  # pin memory for CUDA
    
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
    model = DynamicsMLP(
        state_dim=state_dim,
        action_dim=action_dim,
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
    
    # Learning rate scheduler
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
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_errors = eval_epoch(model, val_loader, criterion, config.device, val_dataset)
        val_losses.append(val_loss)
        
        # Compute component errors
        comp_errors = compute_component_errors(
            val_errors,
            state_dim,
            config.elevation_map_size
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Pos MAE: [{comp_errors['position']['mae'][0]:.4f}, "
                  f"{comp_errors['position']['mae'][1]:.4f}, "
                  f"{comp_errors['position']['mae'][2]:.4f}]")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'hidden_dims': config.hidden_dims,
                    'dropout': config.dropout,
                    'elevation_map_size': config.elevation_map_size,
                    'predict_delta': config.predict_delta,
                },
                'normalization': {
                    'state_mean': train_dataset.state_mean,
                    'state_std': train_dataset.state_std,
                    'action_mean': train_dataset.action_mean,
                    'action_std': train_dataset.action_std,
                    'target_mean': train_dataset.target_mean,
                    'target_std': train_dataset.target_std,
                }
            }
            torch.save(checkpoint, config.save_dir / 'best_model.pt')
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.save_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print("\n" + "=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot training curves
    print("\nGenerating training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.save_dir / 'training_curves.png', dpi=150)
    print(f"Saved: {config.save_dir / 'training_curves.png'}")
    
    # Final evaluation with component breakdown
    print("\nFinal Evaluation")
    print("=" * 80)
    
    model.load_state_dict(torch.load(config.save_dir / 'best_model.pt', weights_only=False)['model_state_dict'])
    _, final_errors = eval_epoch(model, val_loader, criterion, config.device, val_dataset)
    final_comp_errors = compute_component_errors(
        final_errors,
        state_dim,
        config.elevation_map_size
    )
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline dynamics model")
    parser.add_argument('--data_file', type=str, help='Path to H5 data file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    args = parser.parse_args()
    
    main(args)