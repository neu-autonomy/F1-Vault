"""
LSTM variant of the dynamics model (March experiment).

STALE -- predates two fixes that train_cnn.py has: it hard-codes the wrong
elevation layout (676/26x26; the data actually stores 625/25x25 as the last
625 dims, see CLAUDE.md) and does not apply the boundary filter. Port both
fixes from train_cnn.py before training with this again. Note the data's
episodes are mostly 1-2 steps, so sequence_length=5 sequences barely exist.
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
    
    # Sequence parameters
    sequence_length = 5  # Use last 5 timesteps (0.5 seconds of history)
    
    # Model architecture
    hidden_size = 256  # LSTM hidden state size
    num_layers = 2     # Number of LSTM layers
    dropout = 0.2
    
    # Training
    batch_size = 256   # Smaller batch for sequences (more memory)
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 50
    val_split = 0.2
    
    # Prediction target
    predict_delta = True  # Predict change in state
    
    # Preprocessing
    normalize_states = True
    normalize_actions = True
    handle_inf_elevation = True
    
    # Output
    save_dir = Path('models/rnn_dynamics')
    checkpoint_every = 10
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


# ============================================================================
# DATASET (Sequence-based)
# ============================================================================

class SequenceDynamicsDataset(Dataset):
    """Dataset that creates sequences for RNN training."""
    
    def __init__(self, states, actions, next_states, episode_ids, config):
        """
        Args:
            states: (N, state_dim) current states
            actions: (N, action_dim) actions taken
            next_states: (N, state_dim) resulting next states
            episode_ids: (N,) which episode each transition belongs to
            config: Config object
        """
        self.config = config
        self.sequence_length = config.sequence_length
        
        # Handle inf values
        if config.handle_inf_elevation:
            states = self._handle_inf(states.copy())
            next_states = self._handle_inf(next_states.copy())
        
        # Compute deltas if needed
        if config.predict_delta:
            self.targets = next_states - states
        else:
            self.targets = next_states
        
        # Create sequences
        print(f"  Creating sequences of length {self.sequence_length}...")
        self.sequences = self._create_sequences(states, actions, self.targets, episode_ids)
        
        print(f"  Created {len(self.sequences)} sequences from {len(states)} transitions")
        
        # Compute normalization from all data (not sequences)
        self.state_mean = states.mean(axis=0)
        self.state_std = states.std(axis=0) + 1e-8
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-8
        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0) + 1e-8
    
    def _handle_inf(self, data):
        """Replace inf values with sentinel."""
        elevation_size = self.config.elevation_map_size
        elevation_maps = data[:, -elevation_size:]
        inf_mask = np.isinf(elevation_maps)
        if inf_mask.any():
            elevation_maps[inf_mask] = -10.0
            data[:, -elevation_size:] = elevation_maps
        return data
    
    def _create_sequences(self, states, actions, targets, episode_ids):
        """Create sequences that don't cross episode boundaries."""
        sequences = []
        
        # Group by episode
        unique_episodes = np.unique(episode_ids)
        
        for ep_id in unique_episodes:
            # Get all transitions in this episode
            ep_mask = (episode_ids == ep_id)
            ep_states = states[ep_mask]
            ep_actions = actions[ep_mask]
            ep_targets = targets[ep_mask]
            
            # Create sequences within this episode
            ep_len = len(ep_states)
            if ep_len < self.sequence_length:
                continue  # Skip episodes too short
            
            # Sliding window
            for i in range(ep_len - self.sequence_length):
                seq_states = ep_states[i:i+self.sequence_length]
                seq_actions = ep_actions[i:i+self.sequence_length]
                target = ep_targets[i+self.sequence_length-1]  # Predict last target
                
                sequences.append({
                    'states': seq_states,    # (seq_len, state_dim)
                    'actions': seq_actions,  # (seq_len, action_dim)
                    'target': target         # (state_dim,)
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        states = seq['states']
        actions = seq['actions']
        target = seq['target']
        
        # Normalize
        if self.config.normalize_states:
            states = (states - self.state_mean) / self.state_std
            target = (target - self.target_mean) / self.target_std
        
        if self.config.normalize_actions:
            actions = (actions - self.action_mean) / self.action_std
        
        return (
            torch.FloatTensor(states),   # (seq_len, state_dim)
            torch.FloatTensor(actions),  # (seq_len, action_dim)
            torch.FloatTensor(target)    # (state_dim,)
        )


# ============================================================================
# MODEL (LSTM)
# ============================================================================

class DynamicsLSTM(nn.Module):
    """LSTM model for dynamics prediction from sequences."""
    
    def __init__(self, state_dim, action_dim, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Concatenate state and action at each timestep
        input_dim = state_dim + action_dim
        
        # LSTM processes sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # (batch, seq, features)
        )
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, state_dim)
        )
        
        print(f"\nModel Architecture:")
        print(f"  Input per timestep: {input_dim} (state + action)")
        print(f"  LSTM: {num_layers} layers, {hidden_size} hidden size")
        print(f"  Output: {state_dim} (state or delta)")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def forward(self, states, actions):
        """
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
        Returns:
            predictions: (batch, state_dim)
        """
        batch_size, seq_len, _ = states.shape
        
        # Concatenate state and action at each timestep
        x = torch.cat([states, actions], dim=-1)  # (batch, seq_len, state_dim+action_dim)
        
        # Process sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_size)
        
        # Use final hidden state for prediction
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Predict delta or next state
        prediction = self.fc(final_hidden)  # (batch, state_dim)
        
        return prediction


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
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
        
        # Gradient clipping (important for RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(states)
    
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, dataset):
    """Evaluate on validation set."""
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
            
            # Denormalize for interpretable errors
            pred_denorm = predictions.cpu().numpy() * dataset.target_std + dataset.target_mean
            target_denorm = targets.cpu().numpy() * dataset.target_std + dataset.target_mean
            
            errors.append(pred_denorm - target_denorm)
    
    avg_loss = total_loss / len(loader.dataset)
    errors = np.concatenate(errors, axis=0)
    
    return avg_loss, errors


def compute_component_errors(errors, state_dim, elevation_map_size):
    """Compute errors for different state components."""
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


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    config = Config()
    
    # Override config with args if provided
    if args.data_file:
        config.data_file = args.data_file
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.sequence_length:
        config.sequence_length = args.sequence_length
    
    print("=" * 80)
    print("RNN/LSTM DYNAMICS MODEL TRAINING")
    print("=" * 80)
    
    # Create output directory
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    with h5py.File(config.data_file, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]
        next_states = f['next_states'][:]
        episode_ids = f['episode_ids'][:]
        terminated = f['terminated'][:]
        truncated = f['truncated'][:]
        
        state_dim = f.attrs['state_dim']
        action_dim = f.attrs['action_dim']
        
        print(f"Loaded {len(states):,} transitions")
        print(f"State dim: {state_dim}")
        print(f"Action dim: {action_dim}")
    
    # Filter out episode boundaries
    print("\n⚠️  Filtering out episode boundaries (resets)...")
    valid_mask = ~(terminated | truncated)
    
    print(f"  Before filtering: {len(states):,} transitions")
    print(f"  After filtering:  {valid_mask.sum():,} transitions ({100*valid_mask.sum()/len(states):.1f}%)")
    
    states = states[valid_mask]
    actions = actions[valid_mask]
    next_states = next_states[valid_mask]
    episode_ids = episode_ids[valid_mask]
    
    # Split by episode (important for sequences!)
    unique_episodes = np.unique(episode_ids)
    n_episodes = len(unique_episodes)
    n_train_eps = int(n_episodes * (1 - config.val_split))
    
    # Shuffle episodes
    np.random.shuffle(unique_episodes)
    train_episodes = set(unique_episodes[:n_train_eps])
    val_episodes = set(unique_episodes[n_train_eps:])
    
    train_mask = np.array([ep in train_episodes for ep in episode_ids])
    val_mask = np.array([ep in val_episodes for ep in episode_ids])
    
    print(f"\nData split (by episode):")
    print(f"  Train: {n_train_eps} episodes ({train_mask.sum():,} transitions)")
    print(f"  Val:   {n_episodes - n_train_eps} episodes ({val_mask.sum():,} transitions)")
    
    # Create datasets
    print("\nCreating sequence datasets...")
    train_dataset = SequenceDynamicsDataset(
        states[train_mask],
        actions[train_mask],
        next_states[train_mask],
        episode_ids[train_mask],
        config
    )
    
    val_dataset = SequenceDynamicsDataset(
        states[val_mask],
        actions[val_mask],
        next_states[val_mask],
        episode_ids[val_mask],
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
    model = DynamicsLSTM(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
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
    print(f"Sequence length: {config.sequence_length} steps (0.{config.sequence_length}s history)")
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
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_layers,
                    'dropout': config.dropout,
                    'sequence_length': config.sequence_length,
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
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Plot training curves
    print("\nGenerating training curves...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=2)
    ax.plot(val_losses, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'RNN Training Progress (seq_len={config.sequence_length})')
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
    
    # Comparison with MLP
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nTo compare with MLP baseline, check if RNN improves position MAE.")
    print("MLP baseline: ~0.017m (1.7cm)")
    print(f"RNN result:   {final_comp_errors['position']['mae'].mean():.4f}m "
          f"({final_comp_errors['position']['mae'].mean()*100:.2f}cm)")
    
    improvement = (1 - final_comp_errors['position']['mae'].mean() / 0.017) * 100
    if improvement > 0:
        print(f"\n✓ RNN improved by {improvement:.1f}%! Temporal patterns matter.")
    else:
        print(f"\n→ RNN similar performance. Dynamics may be Markovian (current state sufficient).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN/LSTM dynamics model")
    parser.add_argument('--data_file', type=str, help='Path to H5 data file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--sequence_length', type=int, help='Length of input sequences (default: 5)')
    args = parser.parse_args()
    
    main(args)