# scripts/test_imports.py
"""Test that all package imports work"""

print("Testing F1-Vault package structure...")

# Test root import
import f1_vault
print(f"✓ f1_vault version: {f1_vault.__version__}")

# Test data module
from f1_vault.data import (
    RobotState, 
    Action, 
    TrainingChunk,
    TerrainDataset,
    create_dataloaders
)
print("✓ f1_vault.data imports work")

# Test empty modules exist
import f1_vault.models
import f1_vault.physics
import f1_vault.losses
import f1_vault.training
import f1_vault.utils
print("✓ All submodules exist")

print("\n✓ All imports successful!")