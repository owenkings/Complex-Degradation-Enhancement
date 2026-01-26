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
from task2.vgg_feature_wrapper import VGG16FeatureWrapper
from task3.decoder import FeatureDecoder
from task3.perceptual import VGGPerceptual

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "CUB-C"))
    # For decoder training, we only need clean images, so corruption type doesn't matter much
    # provided we just use the 'clean' output from dataset.
    parser.add_argument("--corruption", type=str, default="fog") 
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=str(CURRENT_DIR / "checkpoints"))
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--lambda-perc", type=float, default=0.1)
    return parser.parse_args()

def get_transforms():
    # Standard ImageNet normalization
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
    
    # Feature Decoder (Trainable)
    decoder = FeatureDecoder().to(device)
    
    # 3. Optimization
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    l1_criterion = nn.L1Loss()
    perceptual = VGGPerceptual(layer="relu2_2").to(device)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False, ncols=100)
        for batch_idx, (_, clean, _) in enumerate(train_pbar):
            # We only use clean images for training the decoder (Autoencoder style)
            clean = clean.to(device)
            
            # Extract features (No Grad for VGG)
            with torch.no_grad():
                feat_clean = vgg.extract_shallow_features(clean)
            
            # Decode features
            rec_img = decoder(feat_clean)
            rec_img = torch.clamp(rec_img, -5.0, 5.0)
            
            loss_l1 = l1_criterion(rec_img, clean)
            loss_perc = perceptual(rec_img, clean)
            loss = loss_l1 + args.lambda_perc * loss_perc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % args.print_every == 0:
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "perc": f"{loss_perc.item():.4f}"
                })
                # print(
                #     f"  [Epoch {epoch}][{batch_idx}/{len(train_loader)}] "
                #     f"Loss: {loss.item():.6f} | L1: {loss_l1.item():.6f} | Perc: {loss_perc.item():.6f}"
                # )
                
        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.6f}, Time: {time.time() - start_time:.2f}s")
        
        # Validation
        decoder.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False, ncols=100)
        with torch.no_grad():
            for _, clean, _ in val_pbar:
                clean = clean.to(device)
                
                feat_clean = vgg.extract_shallow_features(clean)
                rec_img = decoder(feat_clean)
                rec_img = torch.clamp(rec_img, -5.0, 5.0)
                
                loss_l1 = l1_criterion(rec_img, clean)
                loss_perc = perceptual(rec_img, clean)
                loss = loss_l1 + args.lambda_perc * loss_perc
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.6f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), os.path.join(args.save_dir, "feature_decoder_best.pth"))
            print(f"  Saved best model to {args.save_dir}/feature_decoder_best.pth")
            
        # Save Last
        torch.save(decoder.state_dict(), os.path.join(args.save_dir, "feature_decoder_last.pth"))

if __name__ == "__main__":
    main()
