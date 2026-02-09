"""
F1-Vault Models Module

Neural network architectures for terrain property prediction.
"""

__version__ = "0.1.0"

# TODO: Add model imports as you create them
from .models import TerrainEncoder

__all__ = [
    "TerrainEncoder",
]