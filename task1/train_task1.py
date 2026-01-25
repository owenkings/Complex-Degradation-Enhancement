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
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints_task1_restormer"))
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
    save_dir = Path(args.save_dir)
    log_path = Path(args.log_path) if args.log_path else save_dir / "train_log.csv"
    pretrained_path = Path(args.pretrained) if args.pretrained else None
    resume_path = Path(args.resume) if args.resume else None

    transform = get_transforms()

    train_ds = CubCTrainDataset(
        root=data_root,
        corruption=corruption,
        split="train",
        transform=transform,
    )
    val_ds = CubCTrainDataset(
        root=data_root,
        corruption=corruption,
        split=val_split,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"[INFO] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
    )

    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = args.amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "corruption", "train_l1", "val_psnr", "val_ssim", "lr", "epoch_time_sec"])

    best_val_psnr = 0.0
    start_epoch = 1

    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_psnr = float(ckpt.get("best_val_psnr", 0.0))
        print(f"[INFO] Resumed from {resume_path}, start_epoch={start_epoch}")
    elif not args.skip_pretrained and pretrained_path:
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        model = load_pretrained(model, pretrained_path, map_location="cpu")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        # ------------- 训练阶段 -------------
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=True)
        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = degraded.to(device)
            clean = clean.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(degraded)
                loss = criterion(pred, clean)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()
            should_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            current_loss = loss.item()
            epoch_loss += current_loss * degraded.size(0)
            
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        epoch_loss /= len(train_loader.dataset)
        print(f"[Epoch {epoch:03d}] Train L1 Loss: {epoch_loss:.6f}")

        # ------------- 验证阶段 (PSNR/SSIM 在 CUB-C test split 上) -------------
        model.eval()
        val_psnr_sum, val_ssim_sum, val_count = 0.0, 0.0, 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", leave=False, ncols=100, mininterval=2.0)
            for degraded, clean, _ in val_pbar:
                degraded = degraded.to(device)
                clean = clean.to(device)

                pred = model(degraded)

                psnr, ssim = batch_psnr_ssim(clean, pred)
                bsz = degraded.size(0)
                val_psnr_sum += psnr * bsz
                val_ssim_sum += ssim * bsz
                val_count += bsz

        val_psnr = val_psnr_sum / val_count
        val_ssim = val_ssim_sum / val_count
        print(f"[Epoch {epoch:03d}] Val PSNR: {val_psnr:.3f}, SSIM: {val_ssim:.4f}")

        epoch_time_sec = time.time() - epoch_start
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, corruption, epoch_loss, val_psnr, val_ssim, optimizer.param_groups[0]["lr"], epoch_time_sec])

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = save_dir / "restormer_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] New best model saved to {best_path} (PSNR={best_val_psnr:.3f})")

        ckpt_path = save_dir / f"restormer_epoch{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_l1": epoch_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_val_psnr": best_val_psnr,
        }, ckpt_path)


if __name__ == "__main__":
    main()
