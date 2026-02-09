"""
F1-Vault: Physics-Informed Learning for Robot-Terrain Interaction

A framework for learning terrain properties from LiDAR data using
self-supervised physics-based training.

Submodules:
- data: Data structures, datasets, and loaders
- models: Neural network architectures
- physics: Physics engine wrappers and force calculations
- losses: Loss functions for training
- training: Training loops and utilities
- utils: Helper functions and utilities
"""

__version__ = "0.1.0"

# Import commonly used items for convenience
from f1_vault.data import RobotState, Action, TrainingChunk

__all__ = [
    "RobotState",
    "Action",
    "TrainingChunk",
]