"""
PyTorch CUDA Diagnostic Script
Checks if CUDA is properly configured and available
"""

import sys
print("=" * 80)
print("PyTorch CUDA Diagnostics")
print("=" * 80)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch is not installed!")
    sys.exit(1)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
else:
    print("\n" + "=" * 80)
    print("CUDA IS NOT AVAILABLE")
    print("=" * 80)
    
    # Check if PyTorch was built with CUDA
    print(f"\nPyTorch built with CUDA: {torch.version.cuda is not None}")
    
    if torch.version.cuda is None:
        print("\n⚠️  PROBLEM IDENTIFIED:")
        print("Your PyTorch installation was built WITHOUT CUDA support!")
        print("\nSOLUTION:")
        print("You need to install the CUDA-enabled version of PyTorch.")
        print("\nFor NVIDIA GPUs, uninstall current PyTorch and reinstall:")
        print("  pip uninstall torch torchvision torchaudio")
        print("\nThen install CUDA version (for CUDA 11.8):")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\nOr for CUDA 12.1:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nOr for CUDA 12.4:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print("\nCheck https://pytorch.org/get-started/locally/ for the latest installation commands")
    else:
        print("\n⚠️  PyTorch has CUDA support, but CUDA is not detected.")
        print("Possible issues:")
        print("  1. NVIDIA drivers not installed or outdated")
        print("  2. CUDA toolkit version mismatch")
        print("  3. GPU not recognized by the system")
        print("\nCheck:")
        print("  - Run 'nvidia-smi' in command prompt to verify GPU is detected")
        print("  - Update NVIDIA drivers from nvidia.com")

# Check for other GPU backends
print("\n" + "=" * 80)
print("Other GPU Backends:")
print("=" * 80)
print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")

# Test tensor creation
print("\n" + "=" * 80)
print("Testing Tensor Creation:")
print("=" * 80)

try:
    cpu_tensor = torch.randn(3, 3)
    print(f"✓ CPU tensor created: shape {cpu_tensor.shape}")
except Exception as e:
    print(f"✗ Failed to create CPU tensor: {e}")

if torch.cuda.is_available():
    try:
        cuda_tensor = torch.randn(3, 3, device='cuda')
        print(f"✓ CUDA tensor created: shape {cuda_tensor.shape}, device {cuda_tensor.device}")
    except Exception as e:
        print(f"✗ Failed to create CUDA tensor: {e}")

print("\n" + "=" * 80)