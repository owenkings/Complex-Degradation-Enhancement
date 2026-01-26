import os
import sys
import argparse
import csv
import time
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument("--backend", type=str, default="mamba", choices=["mamba"], help="Backend used for training (Task 2 forces mamba)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints"))
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--alpha-kl", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--beta-ce", type=float, default=0.0)
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
    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    ce_criterion = nn.CrossEntropyLoss()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    log_path = os.path.join(args.save_dir, "train_log.csv")
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch",
            "train_loss_mse",
            "train_loss_kl",
            "train_loss_ce",
            "train_loss_total",
            "val_loss_mse",
            "val_loss_kl",
            "val_loss_ce",
            "val_loss_total"
        ])
    
    for epoch in range(1, args.epochs + 1):
        enhancer.train()
        epoch_loss_mse = 0.0
        epoch_loss_kl = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_total = 0.0
        start_time = time.time()
        
        for batch_idx, (degraded, clean, labels) in enumerate(train_loader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                feat_deg = vgg.extract_shallow_features(degraded)
                feat_clean = vgg.extract_shallow_features(clean)
                logits_clean = vgg.predict_from_features(feat_clean)
            
            feat_enhanced = enhancer(feat_deg)
            
            logits_enh = vgg.predict_from_features(feat_enhanced)
            
            loss_mse = criterion(feat_enhanced, feat_clean)
            p_clean = F.softmax(logits_clean / args.temperature, dim=1)
            logp_enh = F.log_softmax(logits_enh / args.temperature, dim=1)
            loss_kl = kl_criterion(logp_enh, p_clean) * (args.temperature ** 2)
            
            if args.beta_ce > 0:
                loss_ce = ce_criterion(logits_enh, labels)
            else:
                loss_ce = torch.zeros((), device=device)
            
            loss = loss_mse + args.alpha_kl * loss_kl + args.beta_ce * loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss_mse += loss_mse.item()
            epoch_loss_kl += loss_kl.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_total += loss.item()
            
            if batch_idx % args.print_every == 0:
                print(
                    f"  [Epoch {epoch}][{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} | MSE: {loss_mse.item():.6f} | "
                    f"KL: {loss_kl.item():.6f} | CE: {loss_ce.item():.6f}"
                )
                
        num_train_batches = len(train_loader)
        epoch_loss_mse /= num_train_batches
        epoch_loss_kl /= num_train_batches
        epoch_loss_ce /= num_train_batches
        epoch_loss_total /= num_train_batches
        print(
            f"[Epoch {epoch}] Train Loss: {epoch_loss_total:.6f} "
            f"(MSE: {epoch_loss_mse:.6f}, KL: {epoch_loss_kl:.6f}, CE: {epoch_loss_ce:.6f}), "
            f"Time: {time.time() - start_time:.2f}s"
        )
        
        enhancer.eval()
        val_loss_mse = 0.0
        val_loss_kl = 0.0
        val_loss_ce = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for degraded, clean, labels in val_loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                labels = labels.to(device)
                
                feat_deg = vgg.extract_shallow_features(degraded)
                feat_clean = vgg.extract_shallow_features(clean)
                logits_clean = vgg.predict_from_features(feat_clean)
                feat_enhanced = enhancer(feat_deg)
                logits_enh = vgg.predict_from_features(feat_enhanced)
                
                loss_mse = criterion(feat_enhanced, feat_clean)
                p_clean = F.softmax(logits_clean / args.temperature, dim=1)
                logp_enh = F.log_softmax(logits_enh / args.temperature, dim=1)
                loss_kl = kl_criterion(logp_enh, p_clean) * (args.temperature ** 2)
                if args.beta_ce > 0:
                    loss_ce = ce_criterion(logits_enh, labels)
                else:
                    loss_ce = torch.zeros((), device=device)
                loss = loss_mse + args.alpha_kl * loss_kl + args.beta_ce * loss_ce
                
                val_loss_mse += loss_mse.item()
                val_loss_kl += loss_kl.item()
                val_loss_ce += loss_ce.item()
                val_loss_total += loss.item()
        
        num_val_batches = len(val_loader)
        val_loss_mse /= num_val_batches
        val_loss_kl /= num_val_batches
        val_loss_ce /= num_val_batches
        val_loss_total /= num_val_batches
        print(
            f"[Epoch {epoch}] Val Loss: {val_loss_total:.6f} "
            f"(MSE: {val_loss_mse:.6f}, KL: {val_loss_kl:.6f}, CE: {val_loss_ce:.6f})"
        )
        
        log_writer.writerow([
            epoch,
            f"{epoch_loss_mse:.6f}",
            f"{epoch_loss_kl:.6f}",
            f"{epoch_loss_ce:.6f}",
            f"{epoch_loss_total:.6f}",
            f"{val_loss_mse:.6f}",
            f"{val_loss_kl:.6f}",
            f"{val_loss_ce:.6f}",
            f"{val_loss_total:.6f}"
        ])
        log_file.flush()
        
        # Save Best
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_best.pth"))
            print(f"  Saved best model to {args.save_dir}/mamba_enhancer_best.pth")
            
        # Save Last
        torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_last.pth"))
    
    log_file.close()

if __name__ == "__main__":
    main()
