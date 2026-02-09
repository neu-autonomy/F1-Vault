"""
Debug which batches cause NaN
"""

import torch
from f1_vault.models import TerrainEncoder
from f1_vault.data import create_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, _, _ = create_dataloaders(
    'data/raw/dynamics_data_0000.h5',
    batch_size=8,
    num_workers=0
)

# Create model
model = TerrainEncoder()
model = model.to(device)
model.eval()

print("Testing first 50 batches...")

for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 50:
        break
    
    bev_map = batch['bev_map'].to(device)
    
    # Check input
    has_nan_input = torch.isnan(bev_map).any()
    has_inf_input = torch.isinf(bev_map).any()
    
    if has_nan_input or has_inf_input:
        print(f"\n✗ Batch {batch_idx}: BAD INPUT")
        print(f"   NaN: {torch.isnan(bev_map).sum().item()}")
        print(f"   Inf: {torch.isinf(bev_map).sum().item()}")
        print(f"   Min: {bev_map[~torch.isnan(bev_map)].min().item():.3f}")
        print(f"   Max: {bev_map[~torch.isinf(bev_map)].max().item():.3f}")
        continue
    
    # Check for extreme values
    if bev_map.abs().max() > 100:
        print(f"\n⚠ Batch {batch_idx}: EXTREME VALUES")
        print(f"   Range: [{bev_map.min():.3f}, {bev_map.max():.3f}]")
    
    # Test forward pass
    with torch.no_grad():
        try:
            predictions = model(bev_map)
            
            # Check predictions
            for key, tensor in predictions.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"\n✗ Batch {batch_idx}: NaN in {key}")
                    print(f"   Input range: [{bev_map.min():.3f}, {bev_map.max():.3f}]")
                    break
            else:
                if batch_idx % 10 == 0:
                    print(f"✓ Batch {batch_idx}: OK")
        except Exception as e:
            print(f"\n✗ Batch {batch_idx}: EXCEPTION: {e}")

print("\n✓ Batch testing complete")