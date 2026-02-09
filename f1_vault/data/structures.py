"""
Data Structures for F1-Vault HDF5 files
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch


@dataclass
class RobotState:
    """
    Complete robot state at a single timestep
    
    State vector structure (720 dims):
    - root_pos_w: [3] (x, y, z) - position in world frame
    - root_quat_w: [4] (w, x, y, z) - orientation quaternion
    - world_euler_xyz: [3] (roll, pitch, yaw) - REDUNDANT with quat
    - base_lin_vel: [3] (vx, vy, vz) - linear velocity in body frame
    - base_ang_vel: [3] (ωx, ωy, ωz) - angular velocity in body frame
    - root_lin_vel_w: [3] (vx, vy, vz) - linear velocity in world frame
    - root_ang_vel_w: [3] (ωx, ωy, ωz) - angular velocity in world frame
    - last_action: [2] (throttle, steering)
    - joint_pos: [10] - joint positions
    - joint_vel: [10] - joint velocities
    - elevation_map: [676] - 26×26 heightmap (LAST 676 values)
    
    Total: 3 + 4 + 3 + 3 + 3 + 3 + 3 + 2 + 5 + 5 + 676 = 710 dims
    (Note: Your state is 720, so 10 dims are still unaccounted for)
    """
    raw: np.ndarray
    
    # Position and orientation
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    
    # Velocities
    base_lin_vel: np.ndarray  # Body frame
    base_ang_vel: np.ndarray  # Body frame
    root_lin_vel_w: np.ndarray  # World frame
    root_ang_vel_w: np.ndarray  # World frame
    
    # Terrain
    elevation_map: np.ndarray
    
    def __init__(self, raw_state: np.ndarray):
        """Initialize from raw state vector"""
        self.raw = raw_state
        
        # TODO: Verify these indices match your actual data!
        self.root_pos_w = raw_state[0:3]
        self.root_quat_w = raw_state[3:7]      # FIXED: was 4:7
        # Skip euler at 7:10 (redundant)
        self.base_lin_vel = raw_state[10:13]
        self.base_ang_vel = raw_state[13:16]
        self.root_lin_vel_w = raw_state[16:19]
        self.root_ang_vel_w = raw_state[19:22]
        
        self.elevation_map = raw_state[-676:]
    
    @property
    def pose_7d(self) -> np.ndarray:
        """7D pose: [x, y, z, qw, qx, qy, qz]"""
        return np.concatenate([self.root_pos_w, self.root_quat_w])
    
    @property
    def base_velocity_6d(self) -> np.ndarray:
        """6D velocity in body frame: [vx, vy, vz, wx, wy, wz]"""
        return np.concatenate([self.base_lin_vel, self.base_ang_vel])
    
    @property
    def root_velocity_6d(self) -> np.ndarray:
        """6D velocity in world frame: [vx, vy, vz, wx, wy, wz]"""
        return np.concatenate([self.root_lin_vel_w, self.root_ang_vel_w])
    
    @property
    def elevation_map_2d(self) -> np.ndarray:
        """Elevation map as 2D array (26, 26)"""
        return self.elevation_map.reshape(26, 26)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        return torch.from_numpy(self.raw).float()


@dataclass
class Action:
    """
    Robot control action
    
    - throttle: -1.0 to +1.0 (scaled to ±3.0 m/s target velocity)
    - steering: -1.0 to +1.0 (scaled to ±0.488 rad ≈ ±28°)
    """
    raw: np.ndarray
    throttle: float
    steering: float
    
    def __init__(self, raw_action: np.ndarray):
        """Initialize from raw action vector"""
        self.raw = raw_action
        self.throttle = float(raw_action[0])
        self.steering = float(raw_action[1])
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        return torch.from_numpy(self.raw).float()


@dataclass
class Trajectory:
    """
    Sequence of robot states over time
    
    Attributes:
        states: Tuple of RobotState objects
        actions: Tuple of Action objects
        timesteps: Array of time indices
        is_terminated: Episode ended naturally
        is_truncated: Episode ended by time limit
    """
    states: Tuple[RobotState, ...]
    actions: Tuple[Action, ...]
    timesteps: np.ndarray
    is_terminated: bool
    is_truncated: bool
    
    def __len__(self) -> int:
        """Number of states in trajectory"""
        return len(self.states)
    
    @property
    def duration(self) -> float:
        """Duration in seconds (assuming 10Hz)"""
        return len(self.states) / 10.0
    
    @property
    def poses_7d(self) -> np.ndarray:
        """All poses as (N, 7) array"""
        return np.stack([s.pose_7d for s in self.states])
    
    @property
    def positions(self) -> np.ndarray:
        """All positions as (N, 3) array"""
        return np.stack([s.root_pos_w for s in self.states])
    
    @property
    def orientations(self) -> np.ndarray:
        """All orientations as (N, 4) quaternion array"""
        return np.stack([s.root_quat_w for s in self.states])


@dataclass
class TrainingChunk:
    """
    One training sample: chunk extracted from an episode
    
    This is the unit of data used during training.
    Typically 1 second (10 timesteps at 10Hz).
    
    Attributes:
        episode_id: Source episode ID
        chunk_id: Index of this chunk within episode
        initial_state: Starting state
        actions: Control sequence
        gt_trajectory: Ground truth trajectory
        bev_map: Bird's eye view elevation map (26, 26)
    """
    episode_id: int
    chunk_id: int
    initial_state: RobotState
    actions: Tuple[Action, ...]
    gt_trajectory: Trajectory
    bev_map: np.ndarray  # (26, 26)
    
    def __len__(self) -> int:
        """Number of timesteps in chunk"""
        return len(self.actions)
    
    @property
    def duration(self) -> float:
        """Duration in seconds"""
        return len(self.actions) / 10.0
    
    def to_tensors(self) -> dict:
        """Convert to PyTorch tensors for training"""
        return {
            'bev_map': torch.from_numpy(self.bev_map).float().unsqueeze(0),  # (1, 26, 26)
            'initial_state': self.initial_state.to_tensor(),
            'initial_pose': torch.from_numpy(self.initial_state.pose_7d).float(),
            'actions': torch.stack([a.to_tensor() for a in self.actions]),
            'gt_trajectory': torch.from_numpy(self.gt_trajectory.poses_7d).float(),
        }


@dataclass
class TerrainProperties:
    """
    Predicted terrain properties from neural network
    
    All properties are 2D grids of shape (H, W)
    Network predicts these, then forces are computed from them.
    
    Attributes:
        height: Height where terrain forces start (H, W)
        stiffness: Terrain stiffness coefficient (H, W)
        damping: Damping coefficient (H, W)
        friction: Friction coefficient (H, W)
    """
    height: torch.Tensor      # (B, H, W) or (H, W)
    stiffness: torch.Tensor   # (B, H, W) or (H, W)
    damping: torch.Tensor     # (B, H, W) or (H, W)
    friction: torch.Tensor    # (B, H, W) or (H, W)
    
    @property
    def grid_size(self) -> Tuple[int, int]:
        """Return (height, width) of terrain grid"""
        return tuple(self.height.shape[-2:])
    
    @classmethod
    def from_network_output(cls, output: dict) -> 'TerrainProperties':
        """
        Create from neural network output
        
        Args:
            output: Dict with keys 'height', 'stiffness', 'damping', 'friction'
        """
        return cls(
            height=output['height'],
            stiffness=output['stiffness'],
            damping=output['damping'],
            friction=output['friction']
        )


@dataclass
class PredictedTrajectory:
    """
    Trajectory predicted by physics simulation
    
    Attributes:
        poses: (N, 7) poses [x, y, z, qw, qx, qy, qz]
        velocities: (N, 6) velocities [vx, vy, vz, wx, wy, wz]
        timesteps: Time indices
    """
    poses: torch.Tensor       # (N, 7)
    velocities: torch.Tensor  # (N, 6)
    timesteps: np.ndarray
    
    def __len__(self) -> int:
        return len(self.poses)
    
    @property
    def positions(self) -> torch.Tensor:
        """Positions (N, 3)"""
        return self.poses[:, :3]
    
    @property
    def orientations(self) -> torch.Tensor:
        """Quaternions (N, 4)"""
        return self.poses[:, 3:]