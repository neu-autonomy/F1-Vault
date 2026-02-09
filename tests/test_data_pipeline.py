"""
Test the entire data pipeline
"""

from f1_vault.data import (
    RobotState, 
    Action, 
    TrainingChunk,
    TerrainDataset,
    create_dataloaders,
    analyze_dataset_statistics
)


def test_data_pipeline():
    """Test complete data pipeline"""
    
    hdf5_path = 'data/raw/dynamics_data_0000.h5'
    
    print("="*60)
    print("TESTING F1-VAULT DATA PIPELINE")
    print("="*60)
    
    # Test 1: Analyze dataset
    print("\n1. Analyzing dataset...")
    try:
        stats = analyze_dataset_statistics(hdf5_path)
        for key, val in stats.items():
            print(f"  {key}: {val}")
    except Exception as e:
        print(f"✗ Error analyzing dataset: {e}")
        return
    
    # Test 2: Create dataloaders
    print("\n2. Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            hdf5_path,
            batch_size=4,
            num_workers=0,
            random_seed=42
        )
    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Load one batch
    print("\n3. Loading first batch...")
    try:
        batch = next(iter(train_loader))
        print(f"  Batch type: {type(batch)}")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Batch size: {batch['bev_map'].shape[0]}")
    except Exception as e:
        print(f"✗ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Check batch contents
    print(f"\n4. Checking batch contents...")
    try:
        print(f"  BEV map shape: {batch['bev_map'].shape}")
        print(f"  Initial pose shape: {batch['initial_pose'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        print(f"  GT trajectory shape: {batch['gt_trajectory'].shape}")
        print(f"  Episode IDs: {batch['episode_ids']}")
    except Exception as e:
        print(f"✗ Error checking batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Check tensor properties
    print(f"\n5. Checking tensor properties...")
    try:
        for key, tensor in batch.items():
            if key != 'episode_ids':  # Skip list
                print(f"  {key}: dtype={tensor.dtype}, device={tensor.device}")
    except Exception as e:
        print(f"✗ Error checking tensors: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Iterate through multiple batches
    print(f"\n6. Testing iteration...")
    try:
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            print(f"  Batch {i}: bev_map shape {batch['bev_map'].shape}")
    except Exception as e:
        print(f"✗ Error iterating: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour data pipeline is ready!")
    print("\nBatch format:")
    print("  - bev_map: (B, 1, 26, 26) - Elevation maps")
    print("  - initial_pose: (B, 7) - Starting poses")
    print("  - actions: (B, T, 2) - Control sequences")
    print("  - gt_trajectory: (B, T, 7) - Ground truth trajectories")
    print("  - episode_ids: (B,) - Source episode IDs")
    print("\nNext steps:")
    print("1. Implement terrain encoder model")
    print("2. Implement physics engine wrapper")
    print("3. Implement training loop")


if __name__ == "__main__":
    print("Starting test...")
    test_data_pipeline()
    print("\nTest complete.")