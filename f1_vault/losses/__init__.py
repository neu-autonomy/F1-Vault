"""
F1-Vault Losses Module

Loss functions for training terrain property prediction.
"""

__version__ = "0.1.0"

from .loss import LossWeights, PhysicsInformedTerrainLoss
__all__ = [LossWeights, PhysicsInformedTerrainLoss]