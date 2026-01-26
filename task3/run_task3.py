import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
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
    parser.add_argument("--severity", type=int, default=5, help="Severity level for ImageNet-C (1-5)")
    parser.add_argument("--synset-mapping", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C" / "synset_mapping.txt"), help="Path to synset mapping for ImageNet-C")
    parser.add_argument("--enhancer-ckpt", type=str, default=str(PROJECT_ROOT / "task2" / "checkpoints" / "mamba_enhancer_best.pth"))
    parser.add_argument("--decoder-ckpt", type=str, default=str(CURRENT_DIR / "checkpoints" / "feature_decoder_best.pth"))
    parser.add_argument("--backend", type=str, default="mamba")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-results", action="store_true", help="Save enhanced images")
    parser.add_argument("--output-dir", type=str, default=str(CURRENT_DIR / "results"))
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
    
    try:
        enhancer = MambaFeatureEnhancer(in_channels=128, d_state=16, n_layers=2, backend=args.backend).to(device)
        if os.path.exists(args.enhancer_ckpt):
            enhancer.load_state_dict(torch.load(args.enhancer_ckpt, map_location=device))
            print(f"[INFO] Loaded Enhancer from {args.enhancer_ckpt}")
        else:
            print(f"[WARNING] Enhancer checkpoint not found at {args.enhancer_ckpt}. Using random init.")
        enhancer.eval()
    except Exception as e:
        print(f"[WARNING] Failed to initialize Enhancer ({e}). Using Identity (No Enhancement).")
        enhancer = nn.Identity()
    
    decoder = FeatureDecoder().to(device)
    if os.path.exists(args.decoder_ckpt):
        decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
        print(f"[INFO] Loaded Decoder from {args.decoder_ckpt}")
    else:
        print(f"[WARNING] Decoder checkpoint not found at {args.decoder_ckpt}. Using random init.")
    decoder.eval()
    
    # 2. Dataset
    transform = get_transforms()
    
    if args.dataset_type == "cub-c":
        print(f"[INFO] Loading CUB-C dataset from {args.data_root}")
        test_ds = CubCTrainDataset(
            root=args.data_root,
            corruption=args.corruption,
            split="test",
            transform=transform
        )
    elif args.dataset_type == "imagenet-c":
        print(f"[INFO] Loading ImageNet-C dataset from {args.data_root} (Severity: {args.severity})")
        if not os.path.exists(args.synset_mapping):
            print(f"[ERROR] Synset mapping not found at {args.synset_mapping}")
            return
        synset_mapping = load_synset_mapping(args.synset_mapping)
        test_ds = ImageNetCDataset(
            root=args.data_root,
            corruption=args.corruption,
            severity=args.severity,
            transform=transform,
            synset_mapping=synset_mapping,
            max_samples=1000 # Limit samples for quick inference if needed, or remove for full
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"[INFO] Test samples: {len(test_ds)}")
    
    # 3. Inference
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        
    test_pbar = tqdm(test_loader, desc=f"Run Task3 ({args.corruption})", leave=False, ncols=100)
    with torch.no_grad():
        for i, (degraded, clean, _) in enumerate(test_pbar):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            # 1. Extract Features
            feat_deg = vgg.extract_shallow_features(degraded)
            
            # 2. Enhance Features
            feat_enh = enhancer(feat_deg)
            
            # 3. Decode Features to Image
            img_enh_norm = decoder(feat_enh)
            
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
            total_psnr += psnr
            total_ssim += ssim
            count += 1
            
            # 6. Save first batch results
            if args.save_results and i == 0:
                # Concat: Degraded | Enhanced | Clean
                comparison = torch.cat([img_deg, img_enh, img_clean], dim=3) # Concat horizontally
                save_path = os.path.join(args.output_dir, f"comparison_{args.corruption}.png")
                save_image(comparison, save_path)
                print(f"[INFO] Saved comparison to {save_path}")
                
    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    
    print(f"\n[Results] Corruption: {args.corruption}")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
