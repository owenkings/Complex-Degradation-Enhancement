import torch
import importlib
import sys
import os

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

def check_lib(name):
    try:
        importlib.import_module(name)
        print(f"[OK] {name} imported.")
    except ImportError:
        print(f"[FAIL] {name} not found.")

def check_mamba():
    try:
        from mamba_ssm import Mamba
        print("[OK] mamba_ssm imported.")
    except ImportError:
        print("[FAIL] mamba_ssm not found (Task 2 will use Transformer fallback).")
    except Exception as e:
        print(f"[FAIL] mamba_ssm error: {e}")

if __name__ == "__main__":
    print("Checking environment...")
    check_cuda()
    print("-" * 20)
    for lib in ["torchvision", "cv2", "skimage", "basicsr", "tqdm", "numpy"]:
        # basicsr is inside Restormer, need to check if it's in path or installed
        # In this project, it is in Restormer/basicsr
        # We can try to import it if we add Restormer to path, or check if installed
        check_lib(lib)
    
    print("-" * 20)
    check_mamba()
