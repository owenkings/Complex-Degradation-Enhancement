import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    out_path = Path(args.out_path)

    epochs = []
    train_l1 = []
    val_psnr = []
    val_ssim = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_key = "train_l1" if "train_l1" in row else "train_loss"
            train_l1.append(float(row[train_key]))
            val_psnr.append(float(row["val_psnr"]))
            val_ssim.append(float(row["val_ssim"]))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_l1, label="Train L1", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train L1")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_psnr, label="Val PSNR", color="tab:green")
    ax2.plot(epochs, val_ssim, label="Val SSIM", color="tab:red")
    ax2.set_ylabel("Val PSNR/SSIM")
    ax2.tick_params(axis="y")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)


if __name__ == "__main__":
    main()
