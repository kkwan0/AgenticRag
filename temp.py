import torch
import subprocess

print("="*60)
print("GPU DIAGNOSTICS")
print("="*60)

if torch.cuda.is_available():
    # Basic info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_memory:.2f} GB")
    
    # Current usage
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total_memory - allocated
    
    print(f"\nCurrent Usage:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Free:      {free:.2f} GB")
    
    # Try to see what's using GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("\nnvidia-smi output:")
        print(result.stdout)
    except:
        print("\n(nvidia-smi not available)")
else:
    print("❌ No CUDA available")