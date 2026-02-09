"""
F1-Vault Data Module
"""

__version__ = "0.1.0"

# Import structures
from .structures import (
    RobotState,
    Action,
    Trajectory,
    TrainingChunk,
    TerrainProperties,
    PredictedTrajectory,
)

# Import datasets
from .data import (
    TerrainDataset,
    split_episodes,
    analyze_dataset_statistics,
)

# Import loaders
from .loaders import (
    create_dataloaders,
    custom_collate,  # ← ADD THIS
)

__all__ = [
    # Structures
    "RobotState",
    "Action",
    "Trajectory",
    "TrainingChunk",
    "TerrainProperties",
    "PredictedTrajectory",
    
    # Datasets
    "TerrainDataset",
    "split_episodes",
    "analyze_dataset_statistics",
    
    # Loaders
    "create_dataloaders",
    "custom_collate",  # ← ADD THIS
]