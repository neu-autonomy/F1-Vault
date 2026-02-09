"""
DataLoader utilities for F1-Vault
"""

import h5py
import numpy as np
import torch
from typing import Tuple, List
from torch.utils.data import DataLoader

from .data import TerrainDataset, split_episodes
from .structures import TrainingChunk


def custom_collate(batch: List[TrainingChunk]) -> dict:
    """
    Custom collate function for batching TrainingChunk objects
    
    Converts list of TrainingChunk objects into a dictionary of batched tensors.
    
    Args:
        batch: List of TrainingChunk objects (one per sample in batch)
        
    Returns:
        Dictionary with batched tensors:
        - bev_map: (B, 1, 26, 26)
        - initial_pose: (B, 7)
        - actions: (B, T, 2)
        - gt_trajectory: (B, T, 7)
        - episode_ids: (B,)
    """
    # Stack BEV maps
    bev_maps = torch.stack([
        torch.from_numpy(chunk.bev_map).float().unsqueeze(0)
        for chunk in batch
    ])  # (B, 1, 26, 26)
    
    # Stack initial poses
    initial_poses = torch.stack([
        torch.from_numpy(chunk.initial_state.pose_7d).float()
        for chunk in batch
    ])  # (B, 7)
    
    # Stack actions
    actions = torch.stack([
        torch.stack([a.to_tensor() for a in chunk.actions])
        for chunk in batch
    ])  # (B, T, 2)
    
    # Stack ground truth trajectories
    gt_trajectories = torch.stack([
        torch.from_numpy(chunk.gt_trajectory.poses_7d).float()
        for chunk in batch
    ])  # (B, T, 7)
    
    # Episode IDs
    episode_ids = torch.tensor([chunk.episode_id for chunk in batch])  # (B,)
    
    return {
        'bev_map': bev_maps,
        'initial_pose': initial_poses,
        'actions': actions,
        'gt_trajectory': gt_trajectories,
        'episode_ids': episode_ids,
    }


def create_dataloaders(
    hdf5_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    print(f"Creating dataloaders from {hdf5_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Split: {train_ratio}/{val_ratio}/{1-train_ratio-val_ratio}")
    
    # Load all episode IDs
    with h5py.File(hdf5_path, 'r') as f:
        all_episodes = np.unique(f['episode_ids'][:])
    
    print(f"Found {len(all_episodes)} unique episodes")
    
    # Split episodes
    train_eps, val_eps, test_eps = split_episodes(
        all_episodes, 
        train_ratio, 
        val_ratio, 
        random_seed
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TerrainDataset(hdf5_path, list(train_eps))
    val_dataset = TerrainDataset(hdf5_path, list(val_eps))
    test_dataset = TerrainDataset(hdf5_path, list(test_eps))
    
    # Create dataloaders with custom collate
    print("\nCreating dataloaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=custom_collate  # ← ADD THIS!
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=custom_collate  # ← ADD THIS!
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=custom_collate  # ← ADD THIS!
    )
    
    # Print summary
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} chunks → {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} chunks → {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} chunks → {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader