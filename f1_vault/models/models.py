"""
Terrain Encoder Model

Predicts terrain properties from bird's-eye-view elevation maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TerrainEncoder(nn.Module):
    """
    CNN that predicts terrain properties from elevation maps
    
    Architecture:
        Input (B, 1, 26, 26)
        → Conv layers with BatchNorm
        → Separate 1×1 conv heads for each property
        → Apply activation functions
        → Output: 4 property maps (B, 26, 26 each)
    
    Args:
        input_channels: Number of input channels (default: 1)
        hidden_dims: List of hidden layer dimensions (default: [32, 64, 64])
        min_stiffness: Minimum stiffness value in N/m (default: 100.0)
        height_scale: Scale for height offset (default: 0.5)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: list = None,
        min_stiffness: float = 100.0,
        height_scale: float = 0.5,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 64]
        
        self.min_stiffness = min_stiffness
        self.height_scale = height_scale
        
        # Build encoder (feature extractor)
        layers = []
        in_channels = input_channels
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_channels = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output heads (1×1 convolutions for each property)
        final_dim = hidden_dims[-1]
        self.height_head = nn.Conv2d(final_dim, 1, kernel_size=1)
        self.stiffness_head = nn.Conv2d(final_dim, 1, kernel_size=1)
        self.damping_head = nn.Conv2d(final_dim, 1, kernel_size=1)
        self.friction_head = nn.Conv2d(final_dim, 1, kernel_size=1)
    
    def forward(self, elevation_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            elevation_map: (B, 1, 26, 26) elevation map
            
        Returns:
            Dictionary with terrain properties:
            - 'height': (B, 26, 26) - height where forces start
            - 'stiffness': (B, 26, 26) - terrain stiffness (N/m)
            - 'damping': (B, 26, 26) - damping coefficient
            - 'friction': (B, 26, 26) - friction coefficient [0, 2]
        """
        # Extract features
        features = self.encoder(elevation_map)  # (B, 64, 26, 26)
        
        # Apply output heads
        height_offset = self.height_head(features)    # (B, 1, 26, 26)
        stiffness_raw = self.stiffness_head(features) # (B, 1, 26, 26)
        damping_raw = self.damping_head(features)     # (B, 1, 26, 26)
        friction_raw = self.friction_head(features)   # (B, 1, 26, 26)
        
        # Apply activation functions and constraints
        # Height: Small offset from input elevation
        height = (
            elevation_map.squeeze(1) + 
            torch.tanh(height_offset.squeeze(1)) * self.height_scale
        )
        
        # Stiffness: Positive with minimum value
        stiffness = F.softplus(stiffness_raw.squeeze(1)) + self.min_stiffness
        
        # Damping: Positive
        damping = F.softplus(damping_raw.squeeze(1))
        
        # Friction: Bounded [0, 2]
        friction = torch.sigmoid(friction_raw.squeeze(1)) * 2.0
        
        return {
            'height': height,
            'stiffness': stiffness,
            'damping': damping,
            'friction': friction
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)