import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from utils.dataset import CubCTrainDataset
from utils.seed_utils import set_global_seed

ROOT = Path(__file__).resolve().parent
RESTORMER_DIR = ROOT / "Restormer"

import sys

sys.path.append(str(RESTORMER_DIR))
from basicsr.models.archs.restormer_arch import Restormer


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=str(ROOT / "data" / "CUB-C"))
    parser.add_argument("--corruption", type=str, default="all")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top5", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(2025)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    restormer_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    vgg_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    vgg = build_vgg16(device)
    restormer = build_restormer().to(device).eval()
    restormer = load_checkpoint(restormer, args.ckpt, map_location="cpu")
    restormer.to(device).eval()

    overall_total = 0
    overall_top1_base = 0.0
    overall_top5_base = 0.0
    overall_top1_restored = 0.0
    overall_top5_restored = 0.0
    topk = (1, 5) if args.top5 else (1,)

    for corruption in corruptions:
        dataset = CubCTrainDataset(
            root=args.data_root,
            corruption=corruption,
            split=args.split,
            transform=restormer_transform,
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

        pbar = tqdm(loader, desc=f"Eval {corruption}", leave=False, ncols=100)
        with torch.no_grad():
            for degraded, _, label in pbar:
                degraded = degraded.to(device)
                label = label.to(device)

                vgg_input = vgg_preprocess(degraded)
                logits = vgg(vgg_input)
                accs = accuracy_from_logits(logits, label, topk=topk)
                top1_base += accs[0].item()
                if args.top5:
                    top5_base += accs[1].item()

                restored = restormer(degraded).clamp(0.0, 1.0)
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
        print(f"CUB-C ({corruption}, {args.split}) VGG16 Top-1 baseline: {top1_base:.4f}")
        print(f"CUB-C ({corruption}, {args.split}) VGG16 Top-1 restored: {top1_restored:.4f}")
        if args.top5:
            top5_base /= max(total, 1)
            top5_restored /= max(total, 1)
            print(f"CUB-C ({corruption}, {args.split}) VGG16 Top-5 baseline: {top5_base:.4f}")
            print(f"CUB-C ({corruption}, {args.split}) VGG16 Top-5 restored: {top5_restored:.4f}")

        overall_total += total
        overall_top1_base += top1_base * total
        overall_top1_restored += top1_restored * total
        if args.top5:
            overall_top5_base += top5_base * total
            overall_top5_restored += top5_restored * total

    if len(corruptions) > 1:
        overall_top1_base /= max(overall_total, 1)
        overall_top1_restored /= max(overall_total, 1)
        print(f"CUB-C (all, {args.split}) VGG16 Top-1 baseline: {overall_top1_base:.4f}")
        print(f"CUB-C (all, {args.split}) VGG16 Top-1 restored: {overall_top1_restored:.4f}")
        if args.top5:
            overall_top5_base /= max(overall_total, 1)
            overall_top5_restored /= max(overall_total, 1)
            print(f"CUB-C (all, {args.split}) VGG16 Top-5 baseline: {overall_top5_base:.4f}")
            print(f"CUB-C (all, {args.split}) VGG16 Top-5 restored: {overall_top5_restored:.4f}")


if __name__ == "__main__":
    main()
