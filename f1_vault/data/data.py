"""
Dataset classes for F1-Vault
"""

import h5py
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from torch.utils.data import Dataset

from .structures import RobotState, Action, Trajectory, TrainingChunk


def split_episodes(
    all_episode_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split episodes into train/val/test sets"""
    
    np.random.seed(random_seed)
    
   
    episodes = all_episode_ids.copy()
    np.random.shuffle(episodes)
    

    n_total = len(episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    

    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:n_train + n_val]
    test_eps = episodes[n_train + n_val:]
    
    print(f"Split {n_total} episodes:")
    print(f"  Train: {len(train_eps)} ({len(train_eps)/n_total*100:.1f}%)")
    print(f"  Val: {len(val_eps)} ({len(val_eps)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_eps)} ({len(test_eps)/n_total*100:.1f}%)")
    
    return train_eps, val_eps, test_eps


def analyze_dataset_statistics(hdf5_path: str) -> Dict:
    """Analyze dataset and return statistics"""
    
    with h5py.File(hdf5_path, 'r') as f:

        episode_ids = f['episode_ids'][:]
        

        unique_episodes = np.unique(episode_ids)
        n_episodes = len(unique_episodes)
        

        episode_lengths = []
        for ep_id in unique_episodes:
            length = (episode_ids == ep_id).sum()
            episode_lengths.append(length)
        
        episode_lengths = np.array(episode_lengths)
        

        state_dim = f.attrs['state_dim']
        action_dim = f.attrs['action_dim']
        
        stats = {
            'n_episodes': n_episodes,
            'n_transitions': len(episode_ids),
            'mean_episode_length': float(episode_lengths.mean()),
            'median_episode_length': float(np.median(episode_lengths)),
            'min_episode_length': int(episode_lengths.min()),
            'max_episode_length': int(episode_lengths.max()),
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
    
    return stats
class TerrainDataset(Dataset):
    """Dataset for loading training chunks from HDF5"""
    
    def __init__(
        self,
        hdf5_path: str,
        episode_ids: List[int],
        chunk_duration: float = 1.0,
        control_freq: int = 10
    ):
        """
        Initialize dataset
        
        Args:
            hdf5_path: Path to HDF5 file
            episode_ids: List of episode IDs to include
            chunk_duration: Duration of each chunk in seconds
            control_freq: Control frequency in Hz
        """
        self.hdf5_path = hdf5_path
        self.episode_ids = episode_ids
        self.chunk_duration = chunk_duration
        self.control_freq = control_freq
        self.chunk_length = int(chunk_duration * control_freq)
        
        # Validate file exists
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        # Create chunk index
        print(f"Creating chunk index for {len(episode_ids)} episodes...")
        self.chunks = self._create_chunk_index()
        print(f"Created {len(self.chunks)} chunks (chunk_length={self.chunk_length})")
    
    def _create_chunk_index(self) -> List[Tuple[int, int]]:
        """Create index of all valid chunks"""
        
        chunks = []
        skipped_bad_data = 0
        
        with h5py.File(self.hdf5_path, 'r') as f:
            episode_ids_data = f['episode_ids'][:]
            states = f['states'][:]
            
            for ep_id in self.episode_ids:
                # Find all indices for this episode
                ep_mask = (episode_ids_data == ep_id)
                ep_indices = np.where(ep_mask)[0]
                
                # Skip if episode too short
                if len(ep_indices) < self.chunk_length:
                    continue
                
                # Create non-overlapping chunks
                for i in range(0, len(ep_indices) - self.chunk_length + 1, self.chunk_length):
                    start_idx = ep_indices[i]
                    end_idx = start_idx + self.chunk_length
                    
                    # VALIDATE: Check for NaN/Inf in this chunk
                    chunk_states = states[start_idx:end_idx]
                    
                    if np.isnan(chunk_states).any() or np.isinf(chunk_states).any():
                        skipped_bad_data += 1
                        continue  # Skip this chunk
                    
                    # Chunk is valid
                    chunks.append((ep_id, int(start_idx)))
        
        if skipped_bad_data > 0:
            print(f"  ⚠ Skipped {skipped_bad_data} chunks with NaN/Inf values")
        
        return chunks
    
    def __len__(self) -> int:
        """Return number of chunks"""
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> TrainingChunk:
        """Load one training chunk"""
        
        ep_id, start_idx = self.chunks[idx]
        end_idx = start_idx + self.chunk_length
        
        with h5py.File(self.hdf5_path, 'r') as f:
            states_raw = f['states'][start_idx:end_idx]
            actions_raw = f['actions'][start_idx:end_idx]
            next_states_raw = f['next_states'][start_idx:end_idx]
            
            # Create objects
            initial_state = RobotState(states_raw[0])
            
            actions = tuple(Action(a) for a in actions_raw)
            
            gt_states = tuple(RobotState(s) for s in next_states_raw)
            gt_trajectory = Trajectory(
                states=gt_states,
                actions=actions,
                timesteps=np.arange(self.chunk_length),
                is_terminated=False,
                is_truncated=False
            )
            
            chunk = TrainingChunk(
                episode_id=int(ep_id),
                chunk_id=idx,
                initial_state=initial_state,
                actions=actions,
                gt_trajectory=gt_trajectory,
                bev_map=initial_state.elevation_map_2d
            )
        
        return chunk