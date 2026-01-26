import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from utils.seed_utils import set_global_seed
from utils.dataset import CubCTrainDataset, ImageNetCDataset
from utils.metrics import batch_psnr_ssim
from task2.vgg_feature_wrapper import VGG16FeatureWrapper
from task2.mamba_enhancer import MambaFeatureEnhancer
from task3.decoder import FeatureDecoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "CUB-C"))
    parser.add_argument("--dataset-type", type=str, default="cub-c", choices=["cub-c", "imagenet-c"])
    parser.add_argument("--corruption", type=str, default="fog")
    parser.add_argument("--severity", type=str, default="flat", help="Severity level for ImageNet-C (1-5) or 'flat'")
    parser.add_argument("--synset-mapping", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C" / "synset_mapping.txt"), help="Path to synset mapping for ImageNet-C")
    parser.add_argument("--enhancer-ckpt", type=str, default=str(PROJECT_ROOT / "task2" / "checkpoints" / "mamba_enhancer_best.pth"))
    parser.add_argument("--decoder-ckpt", type=str, default=str(CURRENT_DIR / "checkpoints" / "feature_decoder_best.pth"))
    parser.add_argument("--backend", type=str, default="mamba")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-results", action="store_true", help="Save enhanced images")
    parser.add_argument("--output-dir", type=str, default=str(CURRENT_DIR / "results"))
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0 for all)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    return parser.parse_args()

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def denormalize(tensor):
    """
    Reverse ImageNet normalization.
    tensor: (B, C, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

def load_synset_mapping(mapping_path):
    synset_to_idx = {}
    with open(mapping_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            synset = line.split()[0]
            synset_to_idx[synset] = idx
    return synset_to_idx

def main():
    args = parse_args()
    set_global_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    # 1. Models
    print("[INFO] Loading models...")
    vgg = VGG16FeatureWrapper(device=device)
    
    # Strict Enhancer Loading
    try:
        enhancer = MambaFeatureEnhancer(in_channels=128, d_state=16, n_layers=2, backend=args.backend).to(device)
        if os.path.exists(args.enhancer_ckpt):
            enhancer.load_state_dict(torch.load(args.enhancer_ckpt, map_location=device))
            print(f"[INFO] Loaded Enhancer from {args.enhancer_ckpt}")
        else:
            print(f"[ERROR] Enhancer checkpoint not found at {args.enhancer_ckpt}")
            sys.exit(1)
        enhancer.eval()
    except Exception as e:
        print(f"[ERROR] Failed to initialize/load Enhancer: {e}")
        sys.exit(1)
    
    # Strict Decoder Loading
    try:
        decoder = FeatureDecoder().to(device)
        if os.path.exists(args.decoder_ckpt):
            decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
            print(f"[INFO] Loaded Decoder from {args.decoder_ckpt}")
        else:
            print(f"[ERROR] Decoder checkpoint not found at {args.decoder_ckpt}")
            sys.exit(1)
        decoder.eval()
    except Exception as e:
        print(f"[ERROR] Failed to initialize/load Decoder: {e}")
        sys.exit(1)
    
    # 2. Dataset & Inference Loop
    if args.corruption == "all":
        corruption_list = ["fog", "snow", "brightness", "contrast", "motion_blur"]
    else:
        corruption_list = [c.strip() for c in args.corruption.split(",") if c.strip()]

    print(f"[INFO] Running on corruptions: {corruption_list}")
    
    # Load synset mapping once if needed
    synset_mapping = None
    if args.dataset_type == "imagenet-c":
        if not os.path.exists(args.synset_mapping):
            print(f"[ERROR] Synset mapping not found at {args.synset_mapping}")
            return
        synset_mapping = load_synset_mapping(args.synset_mapping)
    
    transform = get_transforms()
    
    for corruption in corruption_list:
        print(f"\n{'='*20} Processing Corruption: {corruption} {'='*20}")
        
        if args.dataset_type == "cub-c":
            print(f"[INFO] Loading CUB-C dataset from {args.data_root}")
            test_ds = CubCTrainDataset(
                root=args.data_root,
                corruption=corruption,
                split="test",
                transform=transform
            )
            # CubCTrainDataset doesn't implement max_samples internally easily without change, 
            # but we can use Subset or just break loop. 
            # Since user asked for param control, if the dataset class doesn't support it, 
            # we can handle it in the loop or wrap it.
            # However, ImageNetCDataset does support it.
            # For CUB-C, let's just use Subset if needed or break early.
            
        elif args.dataset_type == "imagenet-c":
            print(f"[INFO] Loading ImageNet-C dataset from {args.data_root} (Severity: {args.severity})")
            
            # Check for flat structure explicitly for logging
            corr_path = Path(args.data_root) / corruption
            sev_path = corr_path / str(args.severity)
            if not sev_path.exists() and corr_path.exists():
                print(f"[INFO] Flat severity structure detected (no {args.severity} subdir in {corruption})")
            
            test_ds = ImageNetCDataset(
                root=args.data_root,
                corruption=corruption,
                severity=args.severity,
                transform=transform,
                synset_mapping=synset_mapping,
                max_samples=args.max_samples 
            )
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        print(f"[INFO] Test samples: {len(test_ds)}")
        if args.max_samples > 0 and args.dataset_type == "cub-c":
             print(f"[INFO] Will stop after {args.max_samples} samples (CUB-C manual limit)")

        # 3. Inference
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        
        if args.save_results:
            os.makedirs(args.output_dir, exist_ok=True)
            
        test_pbar = tqdm(test_loader, desc=f"Run Task3 ({corruption})", leave=False, ncols=100)
        with torch.no_grad():
            for i, (degraded, clean, _) in enumerate(test_pbar):
                # Manual max_samples check for CUB-C (since it doesn't support it in __init__)
                if args.dataset_type == "cub-c" and args.max_samples > 0 and count >= args.max_samples:
                    break
                
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                # 1. Extract Features
                feat_deg = vgg.extract_shallow_features(degraded)
                
                # 2. Enhance Features
                feat_enh = enhancer(feat_deg)
                
                # 3. Decode Features to Image
                # Note: train_decoder.py uses normalized 'clean' images as target.
                # Thus, decoder output is also normalized (approx mean=0, std=1).
                # We MUST denormalize it to get back to [0, 1] range.
                img_enh_norm = decoder(feat_enh)
                
                # Validation of decoder output range (for the first batch only)
                if i == 0:
                    min_val, max_val = img_enh_norm.min().item(), img_enh_norm.max().item()
                    mean_val, std_val = img_enh_norm.mean().item(), img_enh_norm.std().item()
                    print(f"\n[DEBUG] Decoder Output Stats (First Batch): Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, Std={std_val:.2f}")
                    if 0.0 <= min_val and max_val <= 1.0:
                         print("[INFO] Decoder output looks close to [0, 1]. Denormalization is still applied because training target is normalized.")
                    else:
                         print("[INFO] Decoder output in normalized range; applying denormalization.")

                # 4. Denormalize for Metrics
                img_enh = denormalize(img_enh_norm)
                img_clean = denormalize(clean)
                img_deg = denormalize(degraded)
                
                # Clip to [0, 1]
                img_enh = torch.clamp(img_enh, 0, 1)
                img_clean = torch.clamp(img_clean, 0, 1)
                img_deg = torch.clamp(img_deg, 0, 1)
                
                # 5. Metrics
                psnr, ssim = batch_psnr_ssim(img_clean, img_enh)
                
                # Adjust for batch size if last batch is smaller (batch_psnr_ssim returns avg over batch)
                # Actually batch_psnr_ssim returns scalar average. 
                # We need to weight it by batch size? 
                # No, usually we just sum up scalars and divide by number of batches? 
                # Wait, if batches are different sizes, simple average of averages is slightly off.
                # But standard practice in these scripts often does simple average.
                # Let's improve it: accumulate sum and divide by total samples.
                
                batch_len = degraded.size(0)
                # batch_psnr_ssim returns average PSNR/SSIM for the batch.
                # So total_psnr += psnr * batch_len
                # But existing code was: total_psnr += psnr; count += 1. 
                # If batch size is constant, it's fine. If last batch is small, it's slightly off.
                # I will stick to the previous pattern to minimize logic changes, 
                # OR fix it. Fixing it is better.
                # BUT, previous code used `count += 1` (counting batches).
                # I'll switch to counting samples for accuracy.
                
                total_psnr += psnr * batch_len
                total_ssim += ssim * batch_len
                count += batch_len
                
                # 6. Save first batch results
                if args.save_results and i == 0:
                    # Concat: Degraded | Enhanced | Clean
                    comparison = torch.cat([img_deg, img_enh, img_clean], dim=3) # Concat horizontally
                    
                    tag = f"{args.dataset_type}_{corruption}_sev-{args.severity}_N-{args.max_samples if args.max_samples>0 else 'all'}"
                    save_path = os.path.join(args.output_dir, f"comparison_{tag}.png")
                    save_image(comparison, save_path)
                    print(f"[INFO] Saved comparison to {save_path}")
                    
        avg_psnr = total_psnr / count if count > 0 else 0
        avg_ssim = total_ssim / count if count > 0 else 0
        
        print(f"\n[Results] Corruption: {corruption}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
