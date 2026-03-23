# """
# Dataset classes for F1-Vault
# """

# import h5py
# import numpy as np
# from typing import List, Tuple, Dict
# from pathlib import Path
# from torch.utils.data import Dataset

# from .structures import RobotState, Action, Trajectory, TrainingChunk


# def split_episodes(
#     all_episode_ids: np.ndarray,
#     train_ratio: float = 0.7,
#     val_ratio: float = 0.15,
#     random_seed: int = 42
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Split episodes into train/val/test sets"""
    
#     np.random.seed(random_seed)
    
   
#     episodes = all_episode_ids.copy()
#     np.random.shuffle(episodes)
    

#     n_total = len(episodes)
#     n_train = int(n_total * train_ratio)
#     n_val = int(n_total * val_ratio)
    

#     train_eps = episodes[:n_train]
#     val_eps = episodes[n_train:n_train + n_val]
#     test_eps = episodes[n_train + n_val:]
    
#     print(f"Split {n_total} episodes:")
#     print(f"  Train: {len(train_eps)} ({len(train_eps)/n_total*100:.1f}%)")
#     print(f"  Val: {len(val_eps)} ({len(val_eps)/n_total*100:.1f}%)")
#     print(f"  Test: {len(test_eps)} ({len(test_eps)/n_total*100:.1f}%)")
    
#     return train_eps, val_eps, test_eps


# def analyze_dataset_statistics(hdf5_path: str) -> Dict:
#     """Analyze dataset and return statistics"""
    
#     with h5py.File(hdf5_path, 'r') as f:

#         episode_ids = f['episode_ids'][:]
        

#         unique_episodes = np.unique(episode_ids)
#         n_episodes = len(unique_episodes)
        

#         episode_lengths = []
#         for ep_id in unique_episodes:
#             length = (episode_ids == ep_id).sum()
#             episode_lengths.append(length)
        
#         episode_lengths = np.array(episode_lengths)
        

#         state_dim = f.attrs['state_dim']
#         action_dim = f.attrs['action_dim']
        
#         stats = {
#             'n_episodes': n_episodes,
#             'n_transitions': len(episode_ids),
#             'mean_episode_length': float(episode_lengths.mean()),
#             'median_episode_length': float(np.median(episode_lengths)),
#             'min_episode_length': int(episode_lengths.min()),
#             'max_episode_length': int(episode_lengths.max()),
#             'state_dim': state_dim,
#             'action_dim': action_dim,
#         }
    
#     return stats
# class TerrainDataset(Dataset):
#     """Dataset for loading training chunks from HDF5"""
    
#     def __init__(
#         self,
#         hdf5_path: str,
#         episode_ids: List[int],
#         chunk_duration: float = 1.0,
#         control_freq: int = 10
#     ):
#         """
#         Initialize dataset
        
#         Args:
#             hdf5_path: Path to HDF5 file
#             episode_ids: List of episode IDs to include
#             chunk_duration: Duration of each chunk in seconds
#             control_freq: Control frequency in Hz
#         """
#         self.hdf5_path = hdf5_path
#         self.episode_ids = episode_ids
#         self.chunk_duration = chunk_duration
#         self.control_freq = control_freq
#         self.chunk_length = int(chunk_duration * control_freq)
        
#         # Validate file exists
#         if not Path(hdf5_path).exists():
#             raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
#         # Create chunk index
#         print(f"Creating chunk index for {len(episode_ids)} episodes...")
#         self.chunks = self._create_chunk_index()
#         print(f"Created {len(self.chunks)} chunks (chunk_length={self.chunk_length})")
    
#     def _create_chunk_index(self) -> List[Tuple[int, int]]:
#         """Create index of all valid chunks"""
        
#         chunks = []
#         skipped_bad_data = 0
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             episode_ids_data = f['episode_ids'][:]
#             states = f['states'][:]
            
#             for ep_id in self.episode_ids:
#                 # Find all indices for this episode
#                 ep_mask = (episode_ids_data == ep_id)
#                 ep_indices = np.where(ep_mask)[0]
                
#                 # Skip if episode too short
#                 if len(ep_indices) < self.chunk_length:
#                     continue
                
#                 # Create non-overlapping chunks
#                 for i in range(0, len(ep_indices) - self.chunk_length + 1, self.chunk_length):
#                     start_idx = ep_indices[i]
#                     end_idx = start_idx + self.chunk_length
                    
#                     # VALIDATE: Check for NaN/Inf in this chunk
#                     chunk_states = states[start_idx:end_idx]
                    
#                     if np.isnan(chunk_states).any() or np.isinf(chunk_states).any():
#                         skipped_bad_data += 1
#                         continue  # Skip this chunk
                    
#                     # Chunk is valid
#                     chunks.append((ep_id, int(start_idx)))
        
#         if skipped_bad_data > 0:
#             print(f"  ⚠ Skipped {skipped_bad_data} chunks with NaN/Inf values")
        
#         return chunks
    
#     def __len__(self) -> int:
#         """Return number of chunks"""
#         return len(self.chunks)
    
#     def __getitem__(self, idx: int) -> TrainingChunk:
#         """Load one training chunk"""
        
#         ep_id, start_idx = self.chunks[idx]
#         end_idx = start_idx + self.chunk_length
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             states_raw = f['states'][start_idx:end_idx]
#             actions_raw = f['actions'][start_idx:end_idx]
#             next_states_raw = f['next_states'][start_idx:end_idx]
            
#             # Create objects
#             initial_state = RobotState(states_raw[0])
            
#             actions = tuple(Action(a) for a in actions_raw)
            
#             gt_states = tuple(RobotState(s) for s in next_states_raw)
#             gt_trajectory = Trajectory(
#                 states=gt_states,
#                 actions=actions,
#                 timesteps=np.arange(self.chunk_length),
#                 is_terminated=False,
#                 is_truncated=False
#             )
            
#             chunk = TrainingChunk(
#                 episode_id=int(ep_id),
#                 chunk_id=idx,
#                 initial_state=initial_state,
#                 actions=actions,
#                 gt_trajectory=gt_trajectory,
#                 bev_map=initial_state.elevation_map_2d
#             )
        
#         return chunk
"""
Dataset classes for F1-Vault
"""

import h5py
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from torch.utils.data import Dataset

from .structures import RobotState, Action, Trajectory, TrainingChunk


# Large multiplier used to build a stable integer trajectory key from
# per-environment episode counters stored in the HDF5.
_EPISODE_KEY_SCALE = 1_000_000


def make_trajectory_keys(env_ids: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Create a globally unique trajectory key for each (env_id, episode_id) pair."""
    env_ids = np.asarray(env_ids, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    return env_ids * _EPISODE_KEY_SCALE + episode_ids


def split_episodes(
    all_episode_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split episodes/trajectories into train/val/test sets."""

    rng = np.random.default_rng(random_seed)
    episodes = np.asarray(all_episode_ids, dtype=np.int64).copy()
    rng.shuffle(episodes)

    n_total = len(episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_eps = episodes[:n_train]
    val_eps = episodes[n_train:n_train + n_val]
    test_eps = episodes[n_train + n_val:]

    print(f"Split {n_total} episodes:")
    print(f"  Train: {len(train_eps)} ({len(train_eps)/max(n_total,1)*100:.1f}%)")
    print(f"  Val: {len(val_eps)} ({len(val_eps)/max(n_total,1)*100:.1f}%)")
    print(f"  Test: {len(test_eps)} ({len(test_eps)/max(n_total,1)*100:.1f}%)")

    return train_eps, val_eps, test_eps



def analyze_dataset_statistics(hdf5_path: str) -> Dict:
    """Analyze dataset and return statistics using unique (env_id, episode_id) trajectories."""

    with h5py.File(hdf5_path, 'r') as f:
        episode_ids = f['episode_ids'][:].astype(np.int64)
        env_ids = f['env_ids'][:].astype(np.int64) if 'env_ids' in f else np.zeros_like(episode_ids)
        timesteps = f['timesteps'][:].astype(np.int64) if 'timesteps' in f else None

        trajectory_keys = make_trajectory_keys(env_ids, episode_ids)
        unique_trajectories = np.unique(trajectory_keys)

        episode_lengths = []
        for traj_key in unique_trajectories:
            mask = trajectory_keys == traj_key
            if timesteps is None:
                length = int(mask.sum())
            else:
                traj_timesteps = timesteps[mask]
                # Count unique timesteps in case duplicates slipped in.
                length = int(np.unique(traj_timesteps).shape[0])
            episode_lengths.append(length)

        episode_lengths = np.asarray(episode_lengths, dtype=np.int64)

        state_dim = f.attrs['state_dim']
        action_dim = f.attrs['action_dim']
        sim_dt = float(f.attrs['sim_dt']) if 'sim_dt' in f.attrs else None
        decimation = int(f.attrs['decimation']) if 'decimation' in f.attrs else None
        control_dt = (sim_dt * decimation) if (sim_dt is not None and decimation is not None) else None

        stats = {
            'n_episodes': int(len(unique_trajectories)),
            'n_transitions': int(len(episode_ids)),
            'mean_episode_length': float(episode_lengths.mean()) if len(episode_lengths) else 0.0,
            'median_episode_length': float(np.median(episode_lengths)) if len(episode_lengths) else 0.0,
            'min_episode_length': int(episode_lengths.min()) if len(episode_lengths) else 0,
            'max_episode_length': int(episode_lengths.max()) if len(episode_lengths) else 0,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'sim_dt': sim_dt,
            'decimation': decimation,
            'control_dt': control_dt,
            'control_freq': (1.0 / control_dt) if control_dt else None,
        }

    return stats


class TerrainDataset(Dataset):
    """Dataset for loading training chunks from HDF5."""

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
            episode_ids: List of globally unique trajectory IDs to include
            chunk_duration: Duration of each chunk in seconds
            control_freq: Control frequency in Hz
        """
        self.hdf5_path = hdf5_path
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.chunk_duration = chunk_duration
        self.control_freq = control_freq
        self.chunk_length = int(round(chunk_duration * control_freq))

        if self.chunk_length <= 0:
            raise ValueError(f"chunk_length must be positive, got {self.chunk_length}")

        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        print(f"Creating chunk index for {len(self.episode_ids)} episodes...")
        self.chunks = self._create_chunk_index()
        print(f"Created {len(self.chunks)} chunks (chunk_length={self.chunk_length})")

    def _create_chunk_index(self) -> List[Tuple[int, int]]:
        """Create index of all valid chunks with strict (env, episode, timestep) continuity."""

        chunks: List[Tuple[int, int]] = []
        skipped_bad_data = 0
        skipped_noncontiguous = 0

        with h5py.File(self.hdf5_path, 'r') as f:
            states = f['states'][:]
            episode_ids_data = f['episode_ids'][:].astype(np.int64)
            env_ids_data = f['env_ids'][:].astype(np.int64) if 'env_ids' in f else np.zeros_like(episode_ids_data)
            timesteps_data = f['timesteps'][:].astype(np.int64) if 'timesteps' in f else np.arange(len(episode_ids_data), dtype=np.int64)
            trajectory_keys = make_trajectory_keys(env_ids_data, episode_ids_data)

            for traj_key in self.episode_ids:
                traj_indices = np.where(trajectory_keys == traj_key)[0]
                if len(traj_indices) < self.chunk_length:
                    continue

                # Sort each trajectory by timestep so chunks are true step-by-step windows.
                order = np.argsort(timesteps_data[traj_indices], kind='stable')
                traj_indices = traj_indices[order]
                traj_timesteps = timesteps_data[traj_indices]

                # Create non-overlapping chunks; each chunk must be contiguous in timestep
                # and clean in state space.
                for i in range(0, len(traj_indices) - self.chunk_length + 1, self.chunk_length):
                    chunk_indices = traj_indices[i:i + self.chunk_length]
                    chunk_timesteps = traj_timesteps[i:i + self.chunk_length]

                    if not np.all(np.diff(chunk_timesteps) == 1):
                        skipped_noncontiguous += 1
                        continue

                    chunk_states = states[chunk_indices]
                    if np.isnan(chunk_states).any() or np.isinf(chunk_states).any():
                        skipped_bad_data += 1
                        continue

                    chunks.append((int(traj_key), int(chunk_indices[0])))

        if skipped_bad_data > 0:
            print(f"  ⚠ Skipped {skipped_bad_data} chunks with NaN/Inf values")
        if skipped_noncontiguous > 0:
            print(f"  ⚠ Skipped {skipped_noncontiguous} non-contiguous chunks")

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> TrainingChunk:
        """Load one training chunk."""

        traj_key, start_idx = self.chunks[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            episode_ids = f['episode_ids'][:].astype(np.int64)
            env_ids = f['env_ids'][:].astype(np.int64) if 'env_ids' in f else np.zeros_like(episode_ids)
            timesteps = f['timesteps'][:].astype(np.int64) if 'timesteps' in f else np.arange(len(episode_ids), dtype=np.int64)
            trajectory_keys = make_trajectory_keys(env_ids, episode_ids)

            start_key = trajectory_keys[start_idx]
            if start_key != traj_key:
                raise RuntimeError(
                    f"Chunk index corruption: expected trajectory {traj_key}, found {start_key} at row {start_idx}"
                )

            traj_indices = np.where(trajectory_keys == traj_key)[0]
            order = np.argsort(timesteps[traj_indices], kind='stable')
            traj_indices = traj_indices[order]

            start_pos = np.where(traj_indices == start_idx)[0]
            if len(start_pos) != 1:
                raise RuntimeError(f"Could not relocate start_idx={start_idx} inside trajectory {traj_key}")
            start_pos = int(start_pos[0])

            chunk_indices = traj_indices[start_pos:start_pos + self.chunk_length]
            if len(chunk_indices) != self.chunk_length:
                raise RuntimeError(
                    f"Chunk at idx={idx} truncated unexpectedly: got {len(chunk_indices)} rows, expected {self.chunk_length}"
                )

            chunk_timesteps = timesteps[chunk_indices]
            if not np.all(np.diff(chunk_timesteps) == 1):
                raise RuntimeError(
                    f"Chunk at idx={idx} is not contiguous in time: {chunk_timesteps.tolist()}"
                )

            states_raw = f['states'][chunk_indices]
            actions_raw = f['actions'][chunk_indices]
            next_states_raw = f['next_states'][chunk_indices]
            terminated = f['terminated'][chunk_indices] if 'terminated' in f else np.zeros(self.chunk_length, dtype=bool)
            truncated = f['truncated'][chunk_indices] if 'truncated' in f else np.zeros(self.chunk_length, dtype=bool)

            initial_state = RobotState(states_raw[0])
            actions = tuple(Action(a) for a in actions_raw)
            gt_states = tuple(RobotState(s) for s in next_states_raw)
            gt_trajectory = Trajectory(
                states=gt_states,
                actions=actions,
                timesteps=chunk_timesteps.copy(),
                is_terminated=bool(np.any(terminated)),
                is_truncated=bool(np.any(truncated)),
            )

            chunk = TrainingChunk(
                episode_id=int(traj_key),
                chunk_id=idx,
                initial_state=initial_state,
                actions=actions,
                gt_trajectory=gt_trajectory,
                bev_map=initial_state.elevation_map_2d
            )

        return chunk