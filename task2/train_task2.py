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
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--val-corruption", type=str, default="fog,contrast,brightness,motion_blur,snow")
    parser.add_argument("--backend", type=str, default="mamba", choices=["mamba"], help="Backend used for training (Task 2 forces mamba)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints"))
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--alpha-kl", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--beta-ce", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def accuracy_from_logits(logits, labels, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
        res.append(correct_k)
    return res

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
        corruption=args.val_corruption,
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
    best_val_top1 = 0.0
    start_epoch = 1
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
            "val_mse",
            "val_loss_kl",
            "val_loss_ce",
            "val_total_loss",
            "val_top1",
            "val_top5"
        ])
    
    resume_ckpt = None
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        resume_ckpt = ckpt if isinstance(ckpt, dict) else None
        if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "optimizer_state_dict" in ckpt):
            if "model_state_dict" in ckpt:
                enhancer.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            if "best_val_loss" in ckpt:
                best_val_loss = float(ckpt["best_val_loss"])
            if "best_val_top1" in ckpt:
                best_val_top1 = float(ckpt["best_val_top1"])
        else:
            enhancer.load_state_dict(ckpt)
        print(f"[INFO] Resumed from {args.resume} (start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}, best_val_top1={best_val_top1:.6f})")

    if start_epoch >= 0:
        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                group["initial_lr"] = group["lr"]

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min, last_epoch=start_epoch - 1)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma, last_epoch=start_epoch - 1)
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_gamma, patience=args.lr_patience)
    
    if scheduler is not None and resume_ckpt is not None and "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])

    for epoch in range(start_epoch, args.epochs + 1):
        enhancer.train()
        epoch_loss_mse = 0.0
        epoch_loss_kl = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_total = 0.0
        start_time = time.time()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False, ncols=100)
        for batch_idx, (degraded, clean, labels) in enumerate(train_pbar):
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
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{loss_mse.item():.4f}",
                    "kl": f"{loss_kl.item():.4f}"
                })
                # print(
                #     f"  [Epoch {epoch}][{batch_idx}/{len(train_loader)}] "
                #     f"Loss: {loss.item():.6f} | MSE: {loss_mse.item():.6f} | "
                #     f"KL: {loss_kl.item():.6f} | CE: {loss_ce.item():.6f}"
                # )
                
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
        val_correct_top1 = 0.0
        val_correct_top5 = 0.0
        val_total = 0
        val_sample_count = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False, ncols=100)
        with torch.no_grad():
            for degraded, clean, labels in val_pbar:
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
                
                batch_size = labels.size(0)
                val_loss_mse += loss_mse.item() * batch_size
                val_loss_kl += loss_kl.item() * batch_size
                val_loss_ce += loss_ce.item() * batch_size
                val_loss_total += loss.item() * batch_size
                val_sample_count += batch_size

                pseudo = logits_clean.argmax(dim=1)
                num_classes = logits_enh.size(1)
                top5_k = 5 if num_classes >= 5 else num_classes
                accs = accuracy_from_logits(logits_enh, pseudo, topk=(1, top5_k))
                val_correct_top1 += float(accs[0].item())
                val_correct_top5 += float(accs[1].item())
                val_total += int(pseudo.size(0))
        
        val_loss_mse /= max(val_sample_count, 1)
        val_loss_kl /= max(val_sample_count, 1)
        val_loss_ce /= max(val_sample_count, 1)
        val_loss_total /= max(val_sample_count, 1)
        val_top1 = val_correct_top1 / max(val_total, 1)
        val_top5 = val_correct_top5 / max(val_total, 1)
        print(
            f"[Epoch {epoch}] Val Loss: {val_loss_total:.6f} "
            f"(MSE: {val_loss_mse:.6f}, KL: {val_loss_kl:.6f}, CE: {val_loss_ce:.6f}) "
            f"Top1: {val_top1:.4f}, Top5: {val_top5:.4f}"
        )
        
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss_total)
            else:
                scheduler.step()
        
        log_writer.writerow([
            epoch,
            f"{epoch_loss_mse:.6f}",
            f"{epoch_loss_kl:.6f}",
            f"{epoch_loss_ce:.6f}",
            f"{epoch_loss_total:.6f}",
            f"{val_loss_mse:.6f}",
            f"{val_loss_kl:.6f}",
            f"{val_loss_ce:.6f}",
            f"{val_loss_total:.6f}",
            f"{val_top1:.6f}",
            f"{val_top5:.6f}"
        ])
        log_file.flush()
        
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_best_top1.pth"))
            print(f"  Saved best top1 model to {args.save_dir}/mamba_enhancer_best_top1.pth")
        
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_best_loss.pth"))
            print(f"  Saved best loss model to {args.save_dir}/mamba_enhancer_best_loss.pth")

        torch.save(enhancer.state_dict(), os.path.join(args.save_dir, "mamba_enhancer_last.pth"))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": enhancer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "best_val_top1": best_val_top1,
            },
            os.path.join(args.save_dir, "mamba_enhancer_last_full.pth"),
        )
    
    log_file.close()

if __name__ == "__main__":
    main()
