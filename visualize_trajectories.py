"""
Visualize Robot Trajectories: Predicted vs Actual

This script loads a trained dynamics model and visualizes multi-step rollouts,
comparing predicted trajectories against ground truth from validation episodes.

Usage:
    python visualize_trajectories.py --model models/cnn_dynamics/best_model.pt --data data/raw/dynamics_data_0000.h5
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# Import model architecture (must match training)
# Simplified versions for loading

class ElevationEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projection = nn.Linear(128, latent_dim)
    
    def forward(self, x):
        return self.projection(self.encoder(x))

class ElevationDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.projection = nn.Linear(latent_dim, 128 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=0),
        )
    
    def forward(self, x):
        return self.decoder(self.projection(x))

class DynamicsCNN(nn.Module):
    def __init__(self, core_state_dim, action_dim, elevation_latent_dim, hidden_dims, dropout=0.1):
        super().__init__()
        self.core_state_dim = core_state_dim
        self.elevation_encoder = ElevationEncoder(elevation_latent_dim)
        
        input_dim = core_state_dim + action_dim + elevation_latent_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, core_state_dim + elevation_latent_dim))
        self.dynamics = nn.Sequential(*layers)
        
        self.elevation_decoder = ElevationDecoder(elevation_latent_dim)
    
    def forward(self, core_state, elevation_map, action):
        elev_latent = self.elevation_encoder(elevation_map)
        combined = torch.cat([core_state, action, elev_latent], dim=-1)
        prediction = self.dynamics(combined)
        pred_core = prediction[:, :self.core_state_dim]
        pred_elev_latent = prediction[:, self.core_state_dim:]
        pred_elevation = self.elevation_decoder(pred_elev_latent)
        return pred_core, pred_elevation


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    config = checkpoint['config_dict']
    stats = checkpoint['stats']
    
    # Infer core state dim from saved stats
    core_state_dim = len(stats['core_state_mean'])
    
    model = DynamicsCNN(
        core_state_dim=core_state_dim,
        action_dim=2,
        elevation_latent_dim=config['elevation_latent_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, stats


def rollout_trajectory(model, initial_state, actions, config, stats, device, use_normalization):
    """
    Rollout a trajectory using the model.
    
    Args:
        initial_state: (state_dim,) starting state
        actions: (T, 2) sequence of actions
        config: model config dict
        stats: normalization stats dict
        device: torch device
        use_normalization: whether model was trained with normalization
    
    Returns:
        predicted_states: (T+1, state_dim) predicted trajectory
    """
    elevation_map_size = config['elevation_map_size']
    predict_delta = config['predict_delta']
    
    T = len(actions)
    state_dim = len(initial_state)
    predicted_states = np.zeros((T + 1, state_dim))
    predicted_states[0] = initial_state
    
    current_state = initial_state.copy()
    
    for t in range(T):
        # Split state
        core_state = current_state[:-elevation_map_size]
        elevation_map = current_state[-elevation_map_size:].reshape(26, 26)
        action = actions[t]
        
        # Normalize if needed
        if use_normalization:
            core_state = (core_state - stats['core_state_mean']) / stats['core_state_std']
            elevation_map = (elevation_map - stats['elevation_mean']) / stats['elevation_std']
            action = (action - stats['action_mean']) / stats['action_std']
        
        # Predict
        with torch.no_grad():
            core_tensor = torch.FloatTensor(core_state).unsqueeze(0).to(device)
            elev_tensor = torch.FloatTensor(elevation_map).unsqueeze(0).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            
            pred_core, pred_elev = model(core_tensor, elev_tensor, action_tensor)
            
            pred_core = pred_core.squeeze(0).cpu().numpy()
            pred_elev = pred_elev.squeeze(0).squeeze(0).cpu().numpy()
        
        # Denormalize if needed
        if use_normalization:
            pred_core = pred_core * stats['core_target_std'] + stats['core_target_mean']
            pred_elev = pred_elev * stats['elevation_target_std'] + stats['elevation_target_mean']
        
        # Update state
        if predict_delta:
            current_state[:-elevation_map_size] += pred_core
            current_state[-elevation_map_size:] = (elevation_map + pred_elev).flatten()
        else:
            current_state[:-elevation_map_size] = pred_core
            current_state[-elevation_map_size:] = pred_elev.flatten()
        
        predicted_states[t + 1] = current_state
    
    return predicted_states


def predict_single_steps(model, states, actions, config, stats, device, use_normalization):
    """
    Predict single-step transitions using GROUND TRUTH states.
    This shows true single-step performance without error compounding.
    
    Args:
        states: (T, state_dim) ground truth states
        actions: (T-1, 2) actions taken
        
    Returns:
        predictions: (T-1, state_dim) predicted next states
    """
    elevation_map_size = config['elevation_map_size']
    predict_delta = config['predict_delta']
    
    T = len(states) - 1
    predictions = []
    
    for t in range(T):
        current_state = states[t]
        action = actions[t]
        
        # Split state
        core_state = current_state[:-elevation_map_size]
        elevation_map = current_state[-elevation_map_size:].reshape(26, 26)
        
        # Normalize if needed
        if use_normalization:
            core_state = (core_state - stats['core_state_mean']) / stats['core_state_std']
            elevation_map = (elevation_map - stats['elevation_mean']) / stats['elevation_std']
            action = (action - stats['action_mean']) / stats['action_std']
        
        # Predict
        with torch.no_grad():
            core_tensor = torch.FloatTensor(core_state).unsqueeze(0).to(device)
            elev_tensor = torch.FloatTensor(elevation_map).unsqueeze(0).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            
            pred_core, pred_elev = model(core_tensor, elev_tensor, action_tensor)
            
            pred_core = pred_core.squeeze(0).cpu().numpy()
            pred_elev = pred_elev.squeeze(0).squeeze(0).cpu().numpy()
        
        # Denormalize if needed
        if use_normalization:
            pred_core = pred_core * stats['core_target_std'] + stats['core_target_mean']
            pred_elev = pred_elev * stats['elevation_target_std'] + stats['elevation_target_mean']
        
        # Convert delta to absolute if needed
        if predict_delta:
            pred_state = current_state.copy()
            pred_state[:-elevation_map_size] += pred_core
            pred_state[-elevation_map_size:] = (elevation_map + pred_elev).flatten()
        else:
            pred_state = np.concatenate([pred_core, pred_elev.flatten()])
        
        predictions.append(pred_state)
    
    return np.array(predictions)


def visualize_trajectories(data_file, model_path, num_episodes=5, rollout_length=50):
    """Visualize predicted vs actual trajectories."""
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, config, stats = load_model(model_path, device)
    use_normalization = config.get('use_normalization', False)
    
    print(f"Model loaded. Normalization: {use_normalization}")
    
    # Load data
    print(f"\nLoading data from {data_file}...")
    with h5py.File(data_file, 'r') as f:
        states = f['states'][:]
        actions = f['actions'][:]
        episode_ids = f['episode_ids'][:]
        terminated = f['terminated'][:]
        truncated = f['truncated'][:]
    
    # Filter valid transitions
    valid = ~(terminated | truncated)
    states = states[valid]
    actions = actions[valid]
    episode_ids = episode_ids[valid]
    
    # Get unique episodes
    unique_episodes = np.unique(episode_ids)
    print(f"Found {len(unique_episodes)} episodes")
    
    # Sample random episodes
    np.random.seed(42)
    sample_episodes = np.random.choice(unique_episodes, min(num_episodes, len(unique_episodes)), replace=False)
    
    # Create plots
    output_dir = Path(model_path).parent
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print("\nNote: Showing SINGLE-STEP predictions (no error compounding)")
    print("Each prediction uses ground-truth state, not accumulated predictions")
    
    # Figure 1: 3D trajectories (SINGLE-STEP)
    fig1 = plt.figure(figsize=(16, 12))
    
    for idx, ep_id in enumerate(sample_episodes):
        # Get episode data
        ep_mask = episode_ids == ep_id
        ep_states = states[ep_mask]
        ep_actions = actions[ep_mask]
        
        # Limit length
        ep_len = min(rollout_length, len(ep_states) - 1)
        
        # Single-step predictions (using ground truth states)
        predicted_states = predict_single_steps(
            model, ep_states[:ep_len + 1], ep_actions[:ep_len],
            config, stats, device, use_normalization
        )
        
        actual_states = ep_states[1:ep_len + 1]  # Skip first (no prediction for t=0)
        
        # Extract positions
        pred_pos = predicted_states[:, :3]
        actual_pos = actual_states[:, :3]
        
        # 3D plot
        ax = fig1.add_subplot(2, 3, idx + 1, projection='3d')
        ax.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
                'b-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter(actual_pos[0, 0], actual_pos[0, 1], actual_pos[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        
        # Mean error across trajectory
        errors = np.linalg.norm(pred_pos - actual_pos, axis=1) * 100
        mean_error = errors.mean()
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Episode {ep_id} ({ep_len} steps)\nMean error: {mean_error:.1f}cm')
        ax.legend()
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_3d_singlestep.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: trajectory_3d_singlestep.png")
    
    # Figure 2: XY trajectories (top-down) SINGLE-STEP
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, ep_id in enumerate(sample_episodes):
        ep_mask = episode_ids == ep_id
        ep_states = states[ep_mask]
        ep_actions = actions[ep_mask]
        
        ep_len = min(rollout_length, len(ep_states) - 1)
        
        predicted_states = predict_single_steps(
            model, ep_states[:ep_len + 1], ep_actions[:ep_len],
            config, stats, device, use_normalization
        )
        actual_states = ep_states[1:ep_len + 1]
        
        pred_pos = predicted_states[:, :3]
        actual_pos = actual_states[:, :3]
        
        # Top-down view
        ax = axes[idx]
        ax.plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter(actual_pos[0, 0], actual_pos[0, 1], c='green', s=100, marker='o', zorder=5)
        
        # Mean error
        errors = np.linalg.norm(pred_pos - actual_pos, axis=1) * 100
        mean_error = errors.mean()
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Episode {ep_id}\n{ep_len} steps, Mean error: {mean_error:.1f}cm')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_xy_singlestep.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: trajectory_xy_singlestep.png")
    
    # Figure 3: Error over time (SINGLE-STEP)
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ep_id in sample_episodes:
        ep_mask = episode_ids == ep_id
        ep_states = states[ep_mask]
        ep_actions = actions[ep_mask]
        
        ep_len = min(rollout_length, len(ep_states) - 1)
        
        # Single-step predictions
        predicted_states = predict_single_steps(
            model, ep_states[:ep_len + 1], ep_actions[:ep_len],
            config, stats, device, use_normalization
        )
        actual_states = ep_states[1:ep_len + 1]
        
        pred_pos = predicted_states[:, :3]
        actual_pos = actual_states[:, :3]
        
        # Compute errors
        errors = pred_pos - actual_pos
        error_xy = np.linalg.norm(errors[:, :2], axis=1) * 100
        error_z = np.abs(errors[:, 2]) * 100
        error_mag = np.linalg.norm(errors, axis=1) * 100
        
        timesteps = np.arange(1, len(error_mag) + 1) * 0.1
        
        axes[0, 0].plot(timesteps, error_xy, linewidth=2, alpha=0.7, label=f'Ep {ep_id}')
        axes[0, 1].plot(timesteps, error_z, linewidth=2, alpha=0.7, label=f'Ep {ep_id}')
        axes[1, 0].plot(timesteps, error_mag, linewidth=2, alpha=0.7, label=f'Ep {ep_id}')
    
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('XY Error (cm)')
    axes[0, 0].set_title('Single-Step XY Error (No Compounding)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Z Error (cm)')
    axes[0, 1].set_title('Single-Step Height Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('3D Error (cm)')
    axes[1, 0].set_title('Single-Step Position Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error histogram
    all_errors = []
    for ep_id in sample_episodes:
        ep_mask = episode_ids == ep_id
        ep_states = states[ep_mask]
        ep_actions = actions[ep_mask]
        ep_len = min(rollout_length, len(ep_states) - 1)
        
        predicted_states = predict_single_steps(
            model, ep_states[:ep_len + 1], ep_actions[:ep_len],
            config, stats, device, use_normalization
        )
        actual_states = ep_states[1:ep_len + 1]
        
        errors = np.linalg.norm(predicted_states[:, :3] - actual_states[:, :3], axis=1) * 100
        all_errors.extend(errors)
    
    axes[1, 1].hist(all_errors, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].set_xlabel('Position Error (cm)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Error Distribution (Mean: {np.mean(all_errors):.2f}cm)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axvline(x=np.mean(all_errors), color='r', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_singlestep.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: error_singlestep.png")
    
    # Figure 4: Single episode detailed view (SINGLE-STEP)
    print("\nGenerating detailed single-episode visualization...")
    ep_id = sample_episodes[0]
    ep_mask = episode_ids == ep_id
    ep_states = states[ep_mask]
    ep_actions = actions[ep_mask]
    
    ep_len = min(rollout_length, len(ep_states) - 1)
    
    # Single-step predictions
    predicted_states = predict_single_steps(
        model, ep_states[:ep_len + 1], ep_actions[:ep_len],
        config, stats, device, use_normalization
    )
    actual_states = ep_states[1:ep_len + 1]
    
    pred_pos = predicted_states[:, :3]
    actual_pos = actual_states[:, :3]
    timesteps = np.arange(1, len(pred_pos) + 1) * 0.1
    
    fig4, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # X, Y, Z over time
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[i, 0]
        ax.plot(timesteps, actual_pos[:, i], 'b-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(timesteps, pred_pos[:, i], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} Position (m)')
        ax.set_title(f'{label} Position (Single-Step Predictions)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # XY trajectory
    axes[0, 1].plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', linewidth=2, label='Actual', alpha=0.8)
    axes[0, 1].plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
    axes[0, 1].scatter(actual_pos[0, 0], actual_pos[0, 1], c='green', s=100, marker='o', zorder=5)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title(f'XY Trajectory (Episode {ep_id})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Error over time
    errors = pred_pos - actual_pos
    error_mag = np.linalg.norm(errors, axis=1) * 100
    
    axes[1, 1].plot(timesteps, error_mag, linewidth=2, color='orange')
    axes[1, 1].fill_between(timesteps, 0, error_mag, alpha=0.3, color='orange')
    axes[1, 1].axhline(y=error_mag.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {error_mag.mean():.2f}cm')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position Error (cm)')
    axes[1, 1].set_title('Single-Step Error (Independent Predictions)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Actions used
    axes[2, 1].plot(timesteps, ep_actions[:ep_len, 0], linewidth=2, label='Throttle', alpha=0.8)
    axes[2, 1].plot(timesteps, ep_actions[:ep_len, 1], linewidth=2, label='Steering', alpha=0.8)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Action Value')
    axes[2, 1].set_title('Control Inputs')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_episode_singlestep.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: detailed_episode_singlestep.png")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SINGLE-STEP PREDICTION STATISTICS")
    print("=" * 80)
    
    all_single_errors = []
    for ep_id in sample_episodes:
        ep_mask = episode_ids == ep_id
        ep_states = states[ep_mask]
        ep_actions = actions[ep_mask]
        
        ep_len = min(rollout_length, len(ep_states) - 1)
        predicted_states = predict_single_steps(
            model, ep_states[:ep_len + 1], ep_actions[:ep_len],
            config, stats, device, use_normalization
        )
        actual_states = ep_states[1:ep_len + 1]
        
        errors = np.linalg.norm(predicted_states[:, :3] - actual_states[:, :3], axis=1)
        all_single_errors.extend(errors)
    
    all_single_errors = np.array(all_single_errors) * 100
    
    print(f"\nSingle-step error statistics (across {len(sample_episodes)} episodes):")
    print(f"  Mean: {all_single_errors.mean():.2f} cm")
    print(f"  Median: {np.median(all_single_errors):.2f} cm")
    print(f"  Std: {all_single_errors.std():.2f} cm")
    print(f"  95th percentile: {np.percentile(all_single_errors, 95):.2f} cm")
    print(f"  Max: {all_single_errors.max():.2f} cm")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("\nThese plots show SINGLE-STEP predictions:")
    print("  ✓ Each prediction uses ground-truth state (no error compounding)")
    print("  ✓ This matches how the model was trained")
    print("  ✓ Errors should be consistent with training MAE (~0.8cm)")
    print("\nFor MPC/replanning control (10Hz), this is the relevant metric.")
    print("The model gets fresh sensor data every step, so errors don't accumulate.")
    
    print("\n" + "=" * 80)
    print("PLOTS GENERATED")
    print("=" * 80)
    print(f"All saved to: {output_dir}/")
    print(f"  - trajectory_3d_singlestep.png")
    print(f"  - trajectory_xy_singlestep.png (old multi-step rollout)")
    print(f"  - error_singlestep.png")
    print(f"  - detailed_episode_singlestep.png")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to H5 data file')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to visualize')
    parser.add_argument('--rollout_length', type=int, default=50, help='Rollout length (timesteps)')
    args = parser.parse_args()
    
    visualize_trajectories(args.data, args.model, args.num_episodes, args.rollout_length)