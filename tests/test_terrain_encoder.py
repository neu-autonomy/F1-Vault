"""
Test terrain encoder model
"""

import torch
from f1_vault.models import TerrainEncoder


def test_model_creation():
    """Test creating model"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    model = TerrainEncoder()
    print("✓ Model created successfully")
    
    # Count parameters
    n_params = model.count_parameters()
    print(f"  Total parameters: {n_params:,}")
    
    # Show architecture
    print(f"\n  Architecture:")
    print(f"    Encoder layers: 3 conv blocks")
    print(f"    Output heads: 4 (height, stiffness, damping, friction)")
    
    assert n_params > 0, "Model has no parameters!"
    print(f"\n✓ Model has {n_params:,} trainable parameters")


def test_forward_pass():
    """Test forward pass with dummy input"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    model = TerrainEncoder()
    model.eval()  # Set to eval mode
    
    # Create dummy input
    batch_size = 4
    elevation_map = torch.randn(batch_size, 1, 26, 26)
    
    print(f"  Input shape: {elevation_map.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(elevation_map)
    
    print("✓ Forward pass successful")
    print(f"  Output type: {type(output)}")
    print(f"  Output keys: {list(output.keys())}")
    
    # Check shapes
    print("\n  Output shapes:")
    for key, tensor in output.items():
        print(f"    {key}: {tensor.shape}")
        expected_shape = (batch_size, 26, 26)
        assert tensor.shape == expected_shape, \
            f"Wrong shape for {key}: expected {expected_shape}, got {tensor.shape}"
    
    print("\n✓ All output shapes correct")


def test_output_ranges():
    """Test that outputs are in reasonable ranges"""
    print("\n" + "="*60)
    print("TEST 3: Output Ranges")
    print("="*60)
    
    model = TerrainEncoder()
    model.eval()
    
    # Dummy input with known range
    elevation_map = torch.randn(2, 1, 26, 26)
    
    with torch.no_grad():
        output = model(elevation_map)
    
    print("\n  Checking output constraints:")
    
    # Height should be close to input
    height_diff = (output['height'] - elevation_map.squeeze(1)).abs()
    print(f"\n  Height:")
    print(f"    Max diff from input: {height_diff.max():.3f}")
    print(f"    Should be ≤ {model.height_scale}")
    assert height_diff.max() <= model.height_scale, "Height offset too large!"
    print(f"    ✓ Height is close to input elevation")
    
    # Stiffness should be positive and > min
    print(f"\n  Stiffness:")
    print(f"    Range: [{output['stiffness'].min():.1f}, {output['stiffness'].max():.1f}] N/m")
    print(f"    Min allowed: {model.min_stiffness} N/m")
    assert output['stiffness'].min() >= model.min_stiffness, "Stiffness below minimum!"
    print(f"    ✓ All stiffness values ≥ {model.min_stiffness}")
    
    # Damping should be positive
    print(f"\n  Damping:")
    print(f"    Range: [{output['damping'].min():.3f}, {output['damping'].max():.3f}]")
    assert output['damping'].min() >= 0, "Damping is negative!"
    print(f"    ✓ All damping values ≥ 0")
    
    # Friction should be [0, 2]
    print(f"\n  Friction:")
    print(f"    Range: [{output['friction'].min():.3f}, {output['friction'].max():.3f}]")
    assert output['friction'].min() >= 0, "Friction is negative!"
    assert output['friction'].max() <= 2.0, "Friction exceeds 2.0!"
    print(f"    ✓ All friction values in [0, 2]")
    
    print("\n✓ All output ranges are valid")


def test_with_real_data():
    """Test with actual data from dataloader"""
    print("\n" + "="*60)
    print("TEST 4: Real Data")
    print("="*60)
    
    try:
        from f1_vault.data import create_dataloaders
    except ImportError:
        print("⚠ Skipping real data test (dataloader not available)")
        return
    
    print("  Loading real data...")
    
    # Load one batch
    train_loader, _, _ = create_dataloaders(
        'data/raw/dynamics_data_0000.h5',
        batch_size=4,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    
    # Create model
    model = TerrainEncoder()
    model.eval()
    
    print(f"  Input shape: {batch['bev_map'].shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch['bev_map'])
    
    print(f"\n  Predictions on real data:")
    for key, tensor in output.items():
        print(f"    {key}:")
        print(f"      Shape: {tensor.shape}")
        print(f"      Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"      Mean: {tensor.mean():.3f}")
        print(f"      Std: {tensor.std():.3f}")
    
    print("\n✓ Model works with real data!")


def test_gradient_flow():
    """Test that gradients flow correctly"""
    print("\n" + "="*60)
    print("TEST 5: Gradient Flow")
    print("="*60)
    
    model = TerrainEncoder()
    model.train()  # Set to training mode
    
    # Dummy input
    elevation_map = torch.randn(2, 1, 26, 26, requires_grad=True)
    
    # Forward pass
    output = model(elevation_map)
    
    # Dummy loss (sum of all outputs)
    loss = (
        output['height'].sum() + 
        output['stiffness'].sum() + 
        output['damping'].sum() + 
        output['friction'].sum()
    )
    
    # Backward pass
    loss.backward()
    
    print("  Checking gradients:")
    
    # Check that all parameters have gradients
    has_grad = 0
    no_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
            grad_norm = param.grad.norm().item()
            print(f"    ✓ {name}: grad_norm = {grad_norm:.6f}")
        else:
            no_grad += 1
            print(f"    ✗ {name}: NO GRADIENT!")
    
    assert no_grad == 0, f"{no_grad} parameters have no gradient!"
    
    print(f"\n✓ All {has_grad} parameters have gradients")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "TERRAIN ENCODER TESTS")
    print("="*70)
    
    test_model_creation()
    test_forward_pass()
    test_output_ranges()
    test_gradient_flow()
    test_with_real_data()
    
    print("\n" + "="*70)
    print(" "*25 + "✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nYour terrain encoder is ready!")
    print("\nNext steps:")
    print("  1. Visualize predictions in notebook")
    print("  2. Implement physics engine wrapper")
    print("  3. Implement loss functions")
    print("  4. Create training loop")