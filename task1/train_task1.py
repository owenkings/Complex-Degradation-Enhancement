import os
import sys
import csv
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root and Restormer to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RESTORMER_DIR = PROJECT_ROOT / "Restormer"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(RESTORMER_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.seed_utils import set_global_seed
from utils.dataset import CubCTrainDataset
from utils.metrics import batch_psnr_ssim
from basicsr.models.archs.restormer_arch import Restormer


ROOT = CURRENT_DIR


def get_transforms():
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),  # [0,1]
    ])


def load_pretrained(model, ckpt_path, map_location="cpu"):
    print(f"[INFO] Loading pretrained weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 兼容不同 ckpt 格式
    if isinstance(ckpt, dict):
        if "params" in ckpt:
            state_dict = ckpt["params"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    print("[INFO] Example keys:", list(state_dict.keys())[:5])

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print("  e.g. missing:", missing[:5])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:5])

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "CUB-C"))
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--val-split", type=str, default="test")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints"))
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=str(RESTORMER_DIR / "Motion_Deblurring" / "pretrained_models" / "motion_deblurring.pth"))
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    corruption = args.corruption
    val_split = args.val_split
    num_epochs = args.epochs
    print_every = args.print_every
    accum_steps = max(1, int(args.accum_steps))

    # 1. Dataset & Dataloader
    print(f"[INFO] Loading CUB-C dataset from {data_root} (split='train')")
    transforms_ = get_transforms()
    train_dataset = CubCTrainDataset(
        root=data_root,
        corruption=corruption,
        split="train",
        transform=transforms_
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[INFO] Loading CUB-C dataset from {data_root} (split='{val_split}')")
    val_dataset = CubCTrainDataset(
        root=data_root,
        corruption=corruption,
        split=val_split,
        transform=transforms_
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model
    # Restormer default args: 
    # inp_channels=3, out_channels=3, dim=48, num_blocks=[4,6,6,8], num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    
    if not args.skip_pretrained and os.path.exists(args.pretrained):
        model = load_pretrained(model, args.pretrained)
    elif not args.skip_pretrained:
        print(f"[WARNING] Pretrained model not found at {args.pretrained}, training from scratch.")
        
    model.to(device)

    # 3. Optimizer & Loss
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume?
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    # 4. Logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = args.log_path if args.log_path else str(save_dir / "train_log.csv")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "time"])

    print(f"[INFO] Start training for {num_epochs} epochs...")

    # 5. Training Loop
    best_psnr = 0.0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Add tqdm for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False, ncols=100)
        
        optimizer.zero_grad()
        
        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = degraded.to(device)
            clean = clean.to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                restored = model(degraded)
                loss = criterion(restored, clean)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accum_steps
            epoch_loss += current_loss
            
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for degraded, clean, _ in val_loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                with torch.cuda.amp.autocast(enabled=args.amp):
                    restored = model(degraded)
                
                # Ensure range [0,1]
                restored = torch.clamp(restored, 0, 1)
                psnr, ssim = batch_psnr_ssim(restored, clean)
                total_psnr += psnr.item() * degraded.size(0)
                total_ssim += ssim.item() * degraded.size(0)

        avg_psnr = total_psnr / len(val_dataset)
        avg_ssim = total_ssim / len(val_dataset)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{num_epochs}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | Time: {epoch_time:.1f}s")

        # Logging
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_psnr, avg_ssim, epoch_time])

        # Checkpointing
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_psnr": best_psnr,
        }
        
        # Save latest
        torch.save(save_dict, save_dir / "restormer_latest.pth")

        # Save best
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(save_dict, save_dir / "restormer_best.pth")
            print(f"[INFO] New best PSNR: {best_psnr:.2f}")

if __name__ == "__main__":
    main()
