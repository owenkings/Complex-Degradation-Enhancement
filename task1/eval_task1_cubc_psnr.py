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
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import CubCTrainDataset
from utils.metrics import batch_psnr_ssim
from utils.seed_utils import set_global_seed
from basicsr.models.archs.restormer_arch import Restormer

ROOT = CURRENT_DIR


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data" / "CUB-C"))
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-json", type=str, default=str(CURRENT_DIR / "logs" / "task1_cubc_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(2025)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    allowed = ["fog", "contrast", "brightness", "motion_blur", "snow", "origin"]
    if args.corruption == "all":
        corruptions = [c for c in allowed if c != "origin"]
    else:
        corruptions = [c.strip() for c in args.corruption.split(",") if c.strip()]
    if not corruptions:
        raise ValueError("No valid corruptions specified")
    for corr in corruptions:
        if corr not in allowed:
            raise ValueError(f"Unsupported corruption '{corr}'")

    model = build_restormer().to(device).eval()
    model = load_checkpoint(model, args.ckpt, map_location="cpu")
    model.to(device).eval()

    results = []
    overall_psnr_sum, overall_ssim_sum, overall_count = 0.0, 0.0, 0
    for corruption in corruptions:
        dataset = CubCTrainDataset(
            root=args.data_root,
            corruption=corruption,
            split=args.split,
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )
        psnr_sum, ssim_sum, count = 0.0, 0.0, 0
        with torch.no_grad():
            for degraded, clean, _ in loader:
                degraded = degraded.to(device)
                clean = clean.to(device)
                restored = model(degraded).clamp(0.0, 1.0)
                psnr, ssim = batch_psnr_ssim(clean, restored)
                bsz = degraded.size(0)
                psnr_sum += psnr * bsz
                ssim_sum += ssim * bsz
                count += bsz
                if args.max_samples and count >= args.max_samples:
                    break
        mean_psnr = psnr_sum / max(count, 1)
        mean_ssim = ssim_sum / max(count, 1)
        overall_psnr_sum += psnr_sum
        overall_ssim_sum += ssim_sum
        overall_count += count
        results.append({
            "corruption": corruption,
            "mean_psnr": float(mean_psnr),
            "mean_ssim": float(mean_ssim),
            "count": int(count),
        })
        print(f"CUB-C ({corruption}, {args.split}) mean PSNR: {mean_psnr:.4f}, mean SSIM: {mean_ssim:.4f}")

    overall_psnr = overall_psnr_sum / max(overall_count, 1)
    overall_ssim = overall_ssim_sum / max(overall_count, 1)
    if len(corruptions) > 1:
        print(f"CUB-C (all, {args.split}) mean PSNR: {overall_psnr:.4f}, mean SSIM: {overall_ssim:.4f}")

    save_path = Path(args.save_json) if args.save_json else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "CUB-C",
            "split": args.split,
            "corruptions": corruptions,
            "mean_psnr": float(overall_psnr),
            "mean_ssim": float(overall_ssim),
            "results": results,
            "ckpt": str(Path(args.ckpt).resolve()),
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
