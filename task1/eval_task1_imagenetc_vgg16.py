import argparse
import json
import sys
from pathlib import Path

# Add project root and Restormer to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
RESTORMER_DIR = PROJECT_ROOT / "Restormer"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(RESTORMER_DIR))

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from utils.seed_utils import set_global_seed
from utils.metrics import batch_psnr_ssim
from basicsr.models.archs.restormer_arch import Restormer

ROOT = CURRENT_DIR


class ImageNetCDataset(Dataset):
    def __init__(self, root, corruption, severity, transform, synset_mapping, max_samples=0):
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.synset_to_idx = synset_mapping
        self.max_samples = max_samples
        self.samples = self._collect_samples()

    def _collect_samples(self):
        if self.corruption == "origin":
            base = self.root / "origin"
        else:
            # Check if severity subdirectory exists
            base_corr = self.root / self.corruption
            base_sev = base_corr / str(self.severity)
            if base_sev.exists():
                base = base_sev
            elif base_corr.exists():
                # Fallback to corruption root if severity folder not found
                # This assumes the corruption folder itself contains the synsets (e.g. flat structure or specific severity download)
                base = base_corr
            else:
                base = base_sev # Let it fail in the next check if neither exists
                
        if not base.exists():
            raise FileNotFoundError(f"ImageNet-C path not found: {base}")

        samples = []
        for synset_dir in base.iterdir():
            if not synset_dir.is_dir():
                continue
            synset = synset_dir.name
            if synset not in self.synset_to_idx:
                continue
            label = self.synset_to_idx[synset]
            for img_path in synset_dir.iterdir():
                if img_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                    continue
                samples.append((img_path, label))
                if self.max_samples and len(samples) >= self.max_samples:
                    return samples

        if len(samples) == 0:
            raise RuntimeError(f"No samples found under {base}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load clean image
        # Assuming structure: root/<corruption>/<severity>/<synset>/<image>
        # Clean image: root/origin/<synset>/<image>
        synset = img_path.parent.name
        filename = img_path.name
        clean_path = self.root / "origin" / synset / filename
        
        if clean_path.exists():
            clean_img = Image.open(clean_path).convert("RGB")
        else:
            # Strict mode: raise error if clean image is missing
            raise FileNotFoundError(f"[ERROR] Clean image not found at {clean_path}. PSNR/SSIM calculation requires paired clean images.")

        if self.transform is not None:
            img = self.transform(img)
            clean_img = self.transform(clean_img)
            
        return img, clean_img, label


def build_restormer():
    return Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )


def load_checkpoint(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    return model


def build_vgg16(device):
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.to(device).eval()
    return vgg


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C"))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--corruptions", type=str, default="fog,motion_blur")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top5", action="store_true")
    parser.add_argument("--save-json", type=str, default=str(CURRENT_DIR / "logs" / "task1_imagenetc_results.json"))
    parser.add_argument("--synset-mapping", type=str, default=str(PROJECT_ROOT / "data" / "ImageNet-C" / "synset_mapping.txt"))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(2025)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    synset_to_idx = load_synset_mapping(args.synset_mapping)
    # TODO: 若标签索引需从 1 开始，请在此处整体 +1

    restormer_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    vgg_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vgg = build_vgg16(device)
    restormer = build_restormer().to(device).eval()
    restormer = load_checkpoint(restormer, args.ckpt, map_location="cpu")
    restormer.to(device).eval()

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = [s.strip() for s in args.severities.split(",") if s.strip()]
    # TODO: 若某些 corruption 没有 severity 目录，请把 severities 改为空并调整逻辑

    results = []
    topk = (1, 5) if args.top5 else (1,)

    for corruption in corruptions:
        for severity in severities:
            dataset = ImageNetCDataset(
                root=args.data_root,
                corruption=corruption,
                severity=severity,
                transform=restormer_transform,
                synset_mapping=synset_to_idx,
                max_samples=args.max_samples,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device == "cuda"),
            )

            total = 0
            top1_base = 0.0
            top5_base = 0.0
            top1_restored = 0.0
            top5_restored = 0.0

            psnr_accum = 0.0
            ssim_accum = 0.0

            pbar = tqdm(loader, desc=f"Eval {corruption} sev {severity}", leave=False, ncols=100)
            with torch.no_grad():
                for img, clean_img, label in pbar:
                    img = img.to(device)
                    clean_img = clean_img.to(device)
                    label = label.to(device)

                    # 1. Inference Restormer
                    # Restormer expects [0,1]. Input img is already [0,1] 256x256.
                    restored = restormer(img)
                    
                    # Clamp to [0,1] just in case
                    restored = torch.clamp(restored, 0, 1)

                    # Calculate PSNR/SSIM
                    batch_psnr, batch_ssim = batch_psnr_ssim(clean_img, restored)
                    psnr_accum += batch_psnr * img.size(0)
                    ssim_accum += batch_ssim * img.size(0)

                    # 2. Inference VGG (Original)
                    vgg_input = vgg_preprocess(img)
                    logits = vgg(vgg_input)
                    accs = accuracy_from_logits(logits, label, topk=topk)
                    top1_base += accs[0].item()
                    if args.top5:
                        top5_base += accs[1].item()

                    # 3. Inference VGG (Restored)
                    vgg_input_restored = vgg_preprocess(restored)
                    logits_restored = vgg(vgg_input_restored)
                    accs_restored = accuracy_from_logits(logits_restored, label, topk=topk)
                    top1_restored += accs_restored[0].item()
                    if args.top5:
                        top5_restored += accs_restored[1].item()

                    total += label.size(0)
                    if args.max_samples and total >= args.max_samples:
                        break

            top1_base /= max(total, 1)
            top1_restored /= max(total, 1)
            avg_psnr = psnr_accum / max(total, 1)
            avg_ssim = ssim_accum / max(total, 1)
            
            row = {
                "corruption": corruption,
                "severity": int(severity),
                "top1_degraded": float(top1_base),
                "top1_restored": float(top1_restored),
                "delta_top1": float(top1_restored - top1_base),
                "psnr": float(avg_psnr),
                "ssim": float(avg_ssim),
            }
            if args.top5:
                top5_base /= max(total, 1)
                top5_restored /= max(total, 1)
                row.update({
                    "top5_degraded": float(top5_base),
                    "top5_restored": float(top5_restored),
                    "delta_top5": float(top5_restored - top5_base),
                })
            results.append(row)

            print(
                f"ImageNet-C {corruption} severity {severity} "
                f"Top-1 degraded={top1_base:.4f}, restored={top1_restored:.4f}, "
                f"delta={top1_restored - top1_base:.4f} | "
                f"PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
            )

    save_path = Path(args.save_json) if args.save_json else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "ImageNet-C",
            "ckpt": str(Path(args.ckpt).resolve()),
            "results": results,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
