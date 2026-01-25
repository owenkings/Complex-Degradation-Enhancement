import os
import sys
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from utils.seed_utils import set_global_seed
from utils.dataset import CubCTrainDataset
from task2.mamba_enhancer import MambaFeatureEnhancer
from task2.vgg_feature_wrapper import VGG16FeatureWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "CUB-C"))
    parser.add_argument("--corruption", type=str, default="all") # Train on all corruptions
    parser.add_argument("--backend", type=str, default="mamba", choices=["mamba", "transformer"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints"))
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()

def get_transforms():
    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # Consistent with Ver2 evaluation: Resize(256) -> CenterCrop(224)
    # Note: For paired training (clean vs degraded), we must ensure spatial alignment.
    # Deterministic transforms (Resize+CenterCrop) are safe. 
    # Random transforms would require fixed seed per pair or joint transform logic.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    args = parse_args()
    set_global_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Device: {device}")
    
    # 1. Dataset
    transform = get_transforms()
    train_ds = CubCTrainDataset(
        root=args.data_root,
        corruption=args.corruption,
        split="train",
        transform=transform
    )
    
    # For validation of Feature Regression, we can use test split of CUB-C
    val_ds = CubCTrainDataset(
        root=args.data_root,
        corruption=args.corruption,
        split="test",
        transform=transform
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # 2. Models
    # VGG Wrapper (Fixed)
    vgg = VGG16FeatureWrapper(device=device)
    # Mamba Enhancer (Trainable)
    # SPL output channels: 128
    enhancer = MambaFeatureEnhancer(in_channels=128, d_state=16, n_layers=2, backend=args.backend).to(device)
    
    # 3. Optimization
    optimizer = torch.optim.AdamW(enhancer.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        enhancer.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (degraded, clean, _) in enumerate(train_loader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # Extract features (No Grad for VGG)
            with torch.no_grad():
                feat_deg = vgg.extract_shallow_features(degraded)
                feat_clean = vgg.extract_shallow_features(clean)
            
            # Enhance features
            feat_enhanced = enhancer(feat_deg)
            
            # Loss: MSE(Enhanced, Clean)
            loss = criterion(feat_enhanced, feat_clean)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % args.print_every == 0:
                print(f"  [Epoch {epoch}][{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
                
        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.6f}, Time: {time.time() - start_time:.2f}s")
        
        # Validation
        enhancer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded, clean, _ in val_loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                feat_deg = vgg.extract_shallow_features(degraded)
                feat_clean = vgg.extract_shallow_features(clean)
                feat_enhanced = enhancer(feat_deg)
                
                loss = criterion(feat_enhanced, feat_clean)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.6f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_best.pth"))
            print(f"  Saved best model to {args.save_dir}/mamba_enhancer_best.pth")
            
        # Save Last
        torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_last.pth"))

if __name__ == "__main__":
    main()
