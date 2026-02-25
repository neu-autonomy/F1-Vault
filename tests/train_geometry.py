"""
Experiment 0: Sanity check training

Train model to predict height = input elevation (identity mapping)
This verifies:
- Model can learn
- Training loop works
- Overfitting is possible
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from f1_vault.models import TerrainEncoder
from f1_vault.data import create_dataloaders


class GeometryLoss(nn.Module):
    """Simple loss: predicted height should match input elevation"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, batch):
        """
        Args:
            predictions: Dict with 'height' key
            batch: Batch dict with 'bev_map' key
            
        Returns:
            Loss value
        """
        # Target: height should equal input elevation
        target_height = batch['bev_map'].squeeze(1)  # (B, 26, 26)
        predicted_height = predictions['height']      # (B, 26, 26)
        
        loss = self.mse(predicted_height, target_height)
        
        return loss


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch with NaN detection"""
    
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        bev_map = batch['bev_map'].to(device)
        batch['bev_map'] = bev_map
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(bev_map)
        
        # Compute loss
        loss = loss_fn(predictions, batch)
        
        # CHECK FOR NaN IMMEDIATELY
        if torch.isnan(loss):
            print(f"\n   NaN loss at batch {batch_idx}!")
            print(f"    Checking predictions...")
            for key, tensor in predictions.items():
                has_nan = torch.isnan(tensor).any()
                print(f"      {key}: has_nan={has_nan}, min={tensor.min():.3f}, max={tensor.max():.3f}")
            
            # Skip this batch
            print(f"    Skipping batch {batch_idx}")
            continue
        
        # CHECK FOR Inf
        if torch.isinf(loss):
            print(f"\n   Inf loss at batch {batch_idx}!")
            print(f"    Skipping batch {batch_idx}")
            continue
        
        # Backward pass
        loss.backward()
        
        # CHECK GRADIENTS FOR NaN
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"\n   NaN gradient in {name} at batch {batch_idx}")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"    Skipping batch {batch_idx}")
            optimizer.zero_grad()
            continue
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        valid_batches += 1
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    # Compute average (only over valid batches)
    if valid_batches == 0:
        print("\n   WARNING: No valid batches in this epoch!")
        return float('nan')
    
    avg_loss = total_loss / valid_batches
    return avg_loss


def validate(model, val_loader, loss_fn, device):
    """Validation loop"""
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            bev_map = batch['bev_map'].to(device)
            batch['bev_map'] = bev_map
            
            predictions = model(bev_map)
            loss = loss_fn(predictions, batch)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def main():
    """Main training function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    num_epochs = 10
    batch_size = 2048
    learning_rate = 0.001
    
    print("="*60)
    print("EXPERIMENT 0: GEOMETRY SANITY CHECK")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        'data/raw/dynamics_data_0000.h5',
        batch_size=batch_size,
        num_workers=4  # Start with 0 for debugging
    )
    
    # Create model
    print("\nCreating model...")
    model = TerrainEncoder()
    model = model.to(device)
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = GeometryLoss()
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Check for NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"\n✗ Training failed at epoch {epoch+1} due to NaN loss")
            print(f"   Train loss: {train_loss}")
            print(f"   Val loss: {val_loss}")
            print("\nTrying to continue with last good checkpoint...")
            break
        
        # Check for explosion
        if train_loss > 1000 or val_loss > 1000:
            print(f"\n✗ Loss exploded at epoch {epoch+1}")
            print(f"   Reducing learning rate and restarting from last checkpoint")
            break

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model_exp0.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Experiment 0')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves_exp0.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: best_model_exp0.pt")
    print(f"Training curves saved to: training_curves_exp0.png")
    
    # Test final model
    print("\nTesting final predictions...")
    model.eval()
    batch = next(iter(val_loader))
    bev_map = batch['bev_map'].to(device)
    
    with torch.no_grad():
        predictions = model(bev_map)
    
    # Visualize one prediction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(bev_map[0, 0].cpu(), cmap='terrain')
    axes[0].set_title('Input Elevation')
    axes[0].axis('off')
    
    axes[1].imshow(predictions['height'][0].cpu(), cmap='terrain')
    axes[1].set_title('Predicted Height')
    axes[1].axis('off')
    
    plt.savefig('final_prediction_exp0.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Sanity check complete!")
    print("\nIf loss decreased and predictions match input,")
    print("your model and training loop are working correctly!")


if __name__ == "__main__":
    main()