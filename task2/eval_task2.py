import os
import sys
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

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
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C"))
    parser.add_argument("--dataset-type", type=str, default="imagenet-c", choices=["imagenet-c", "cub-c"])
    parser.add_argument("--enhancer-path", type=str, default=str(CURRENT_DIR / "checkpoints" / "mamba_enhancer_best.pth"))
    parser.add_argument("--backend", type=str, default="mamba", choices=["mamba"], help="Backend used for training")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--baseline", action="store_true", help="Run without enhancer (baseline)")
    parser.add_argument("--corruption", type=str, default="all", help="Comma separated list of corruptions or 'all'")
    parser.add_argument("--severity", type=str, default="flat", help="Comma separated list of severities (1-5) or 'all' or 'flat'")
    parser.add_argument("--synset-mapping", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C" / "synset_mapping.txt"))
    return parser.parse_args()

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

class ImageNetDataset(ImageFolder):
    def __init__(self, root, transform=None, synset_mapping=None):
        super().__init__(root, transform=transform)
        self.synset_mapping = synset_mapping
        
    def __getitem__(self, index):
        path, _ = self.samples[index]
        # 解析 wnid: 假设结构为 root/wnid/image.jpg
        wnid = Path(path).parent.name
        
        # 加载图像
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        # 使用映射的 label
        if self.synset_mapping and wnid in self.synset_mapping:
            target = self.synset_mapping[wnid]
        else:
            target = self.synset_mapping[wnid] # Will raise KeyError if missing
            
        return sample, target

def get_transforms():
    # Preprocessing for ImageNet-C Ver2 (as requested):
    # Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()
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
    
    # 1. Models
    vgg = VGG16FeatureWrapper(device=device)
    
    enhancer = None
    if not args.baseline:
        print(f"[INFO] Loading Enhancer from {args.enhancer_path} (Backend: {args.backend})")
        enhancer = MambaFeatureEnhancer(in_channels=128, d_state=16, n_layers=2, backend=args.backend).to(device)
        try:
            state_dict = torch.load(args.enhancer_path, map_location=device)
            enhancer.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"[ERROR] Checkpoint not found at {args.enhancer_path}. Please train first or run with --baseline.")
            return
        enhancer.eval()
    else:
        print("[INFO] Running Baseline (No Enhancement)")
        
    # 2. Dataset Logic
    data_root = Path(args.data_root)
    ALL_CORRUPTIONS = [
        "fog", "brightness", "contrast", "defocus_blur", "elastic_transform",
        "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
        "jpeg_compression", "motion_blur", "pixelate", "saturate",
        "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"
    ] # List might be incomplete compared to actual folder, but good enough for now. 
      # Better to listdir? 
      # The 15 common corruptions + extra? 
      # Let's rely on directory existence.
    
    if args.dataset_type == "imagenet-c":
        # Check root
        if not data_root.exists():
            print(f"[ERROR] Data root {data_root} does not exist.")
            return
            
        # Determine corruptions
        if args.corruption == "all":
            # Use fixed list to ensure consistency with Task 1
            corruptions = ["fog", "contrast", "brightness", "motion_blur", "snow"]
        else:
            corruptions = args.corruption.split(",")
            
        # Determine severities
        if args.severity == "all":
            severities = [1, 2, 3, 4, 5]
        elif args.severity == "flat":
            severities = ["flat"]
        else:
            severities = [int(s) for s in args.severity.split(",")]
            
    else:
        # CUB-C
        if args.corruption == "all":
             # CubCTrainDataset supports list of corruptions? No, it takes one str or list.
             # But we want to iterate to report per-corruption metrics?
             # Or just report one big average?
             # Task 1 uses global average. 
             # Let's iterate for detailed report.
             corruptions = ["fog", "contrast", "brightness", "motion_blur", "snow"]
        else:
             corruptions = args.corruption.split(",")
        
        severities = ["combined"] # CUB-C dataset structure abstracts severity or mixes them?
        # CubCTrainDataset mixes severities 1-5 inside each corruption folder?
        # Looking at CubCTrainDataset logic: it reads all files in corruption folder.
        # So severity is implicit.
        
    transform = get_transforms()
    
    # Load synset mapping
    synset_to_idx = load_synset_mapping(args.synset_mapping)

    # 3. Evaluation Loop
    results_list = []
    total_acc = 0.0
    count = 0
    
    # Check if we are running a quick sanity check
    if args.baseline and args.corruption == "origin":
        print("[INFO] Running Sanity Check on Origin Data...")
        corruptions = ["origin"] # Override
        severities = ["flat"]   # Origin has no severity

    for corr in sorted(corruptions):
        for sev in severities:
            if corr == "origin":
                target_dir = data_root / "origin"
                desc = "Origin"
            else:
                if sev == "flat":
                     target_dir = data_root / corr
                     desc = f"{corr}"
                else:
                     target_dir = data_root / corr / str(sev)
                     desc = f"{corr} (sev={sev})"
            
            if not target_dir.exists():
                # Fallback logic: if severity folder missing, try parent
                if sev != "flat" and (data_root / corr).exists():
                     # Check if parent has images directly (flat structure)
                     # But ImageFolder needs subfolders (synsets). 
                     # If (data_root/corr) contains synset folders, then it is flat.
                     target_dir = data_root / corr
                     desc = f"{corr} (fallback flat)"
                else:
                    print(f"[WARN] Skipping {desc}: Path {target_dir} not found")
                    continue
            
            # Use Custom ImageNetDataset with explicit mapping
            try:
                dataset = ImageNetDataset(root=str(target_dir), transform=get_transforms(), synset_mapping=synset_to_idx)
            except Exception as e:
                print(f"[WARN] Failed to load dataset at {target_dir}: {e}")
                continue

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            # Sanity Check Print (First batch only)
            if count == 0:
                 print(f"[Sanity Check] First 3 samples from {desc}:")
                 for i in range(min(3, len(dataset))):
                     path, _ = dataset.samples[i]
                     wnid = Path(path).parent.name
                     label = dataset[i][1]
                     print(f"  Path: {Path(path).name}, WNID: {wnid}, Mapped Label: {label}")

            corr_correct = 0
            corr_total = 0
            
            pbar = tqdm(loader, desc=f"Eval {desc}", leave=False, ncols=100)
            with torch.no_grad():
                for imgs, labels in pbar:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    
                    # Extract Features
                    feats = vgg.extract_shallow_features(imgs)
                    
                    # Enhance (if not baseline)
                    if enhancer:
                        feats = enhancer(feats)
                        
                    # Classify
                    logits = vgg.forward_from_shallow(feats)
                    
                    # Accuracy
                    preds = logits.argmax(dim=1)
                    corr_correct += (preds == labels).sum().item()
                    corr_total += labels.size(0)
            
            if corr_total > 0:
                acc = corr_correct / corr_total
                print(f"[{desc}] Accuracy: {acc:.2%}")
                total_acc += acc
                count += 1
                
                results_list.append({
                    "corruption": corr,
                    "severity": sev,
                    "accuracy": acc,
                    "correct": corr_correct,
                    "total": corr_total
                })
            else:
                 print(f"[{desc}] No samples found.")
                 
    import json

    if count > 0:
        avg_acc = total_acc / count
        print(f"\n[Summary] Average Accuracy: {avg_acc:.2%}")
    else:
        print("\n[Summary] No evaluations performed.")
    
    # Save results to JSON
    save_path = PROJECT_ROOT / "task2" / "logs" / "task2_imagenetc_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "dataset": "ImageNet-C" if args.dataset_type == "imagenet-c" else "CUB-C",
        "enhancer_path": str(Path(args.enhancer_path).resolve()) if args.enhancer_path else "baseline",
        "results": results_list
    }
    
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Results saved to {save_path}")

if __name__ == "__main__":
    main()
