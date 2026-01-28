import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Task 1 (Restormer) and Task 2 (Feature Enhancement) results")
    parser.add_argument("--task1-json", type=str, default=str(PROJECT_ROOT / "task1" / "logs" / "task1_imagenetc_results.json"))
    parser.add_argument("--task2-json", type=str, default=str(PROJECT_ROOT / "task2" / "logs" / "task2_imagenetc_results.json"))
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=str, default=str(PROJECT_ROOT / "task3" / "comparison_results"))
    return parser.parse_args()

def load_json(path):
    if not Path(path).exists():
        print(f"[WARN] File not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t1_data = load_json(args.task1_json)
    t2_data = load_json(args.task2_json)

    if not t1_data and not t2_data:
        print("[ERROR] No data found.")
        return

    records = []

    # Process Task 1 Data
    if t1_data:
        print(f"[INFO] Loaded Task 1 results from {args.task1_json}")
        for item in t1_data.get("results", []):
            records.append({
                "Method": "Task1 (Image Enhancement)",
                "Corruption": item["corruption"],
                "Severity": item["severity"],
                "Top1 Accuracy": item["top1_restored"],
                "Top5 Accuracy": item.get("top5_restored"),
                "PSNR": item.get("psnr", 0),
                "SSIM": item.get("ssim", 0),
                "Baseline Accuracy": item["top1_degraded"]
            })

    # Process Task 2 Data
    if t2_data:
        print(f"[INFO] Loaded Task 2 results from {args.task2_json}")
        for item in t2_data.get("results", []):
            sev = item["severity"]
            # Task 2 might use "flat" or int
            try:
                sev = int(sev)
            except:
                pass
            
            records.append({
                "Method": "Task2 (Feature Enhancement)",
                "Corruption": item["corruption"],
                "Severity": sev,
                "Top1 Accuracy": item["accuracy"],
                "Top5 Accuracy": item.get("top5"),
                "PSNR": None, # Task 2 doesn't produce images
                "SSIM": None,
                "Baseline Accuracy": None # Assuming Task 1 covers baseline, or we could run baseline in Task 2
            })

    if not records:
        print("[WARN] No records extracted.")
        return

    df = pd.DataFrame(records)
    numeric_cols = ["Top1 Accuracy", "Top5 Accuracy", "PSNR", "SSIM", "Baseline Accuracy"]
    
    # Save CSV
    csv_path = output_dir / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Summary saved to {csv_path}")
    
    df_avg = df.groupby(["Method", "Corruption"], as_index=False)[numeric_cols].mean(numeric_only=True)
    df_avg["Severity"] = None
    df_overall = df.groupby(["Method"], as_index=False)[numeric_cols].mean(numeric_only=True)
    df_overall["Corruption"] = "all"
    df_overall["Severity"] = None
    summary_df = pd.concat([df_avg, df_overall], ignore_index=True)
    summary_path = output_dir / "comparison_summary_avg.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Summary (avg) saved to {summary_path}")

    # Plotting
    # 1. Accuracy Comparison by Corruption (Average over severities)
    if "Severity" in df.columns:
        # Filter out 'flat' or non-numeric severities for average calculation if mixed
        # But here we assume consistency.
        pass

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_avg, x="Corruption", y="Top1 Accuracy", hue="Method")
    plt.title("Top-1 Accuracy Comparison: Task 1 vs Task 2")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison_bar.png")
    print(f"[INFO] Plot saved to {output_dir / 'accuracy_comparison_bar.png'}")

    # 2. PSNR/SSIM for Task 1 (if available)
    t1_df = df[df["Method"] == "Task1 (Image Enhancement)"]
    if not t1_df.empty and t1_df["PSNR"].sum() > 0:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=t1_df, x="Corruption", y="PSNR")
        plt.title("Task 1 PSNR by Corruption")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.barplot(data=t1_df, x="Corruption", y="SSIM")
        plt.title("Task 1 SSIM by Corruption")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "task1_quality_metrics.png")
        print(f"[INFO] Plot saved to {output_dir / 'task1_quality_metrics.png'}")

if __name__ == "__main__":
    main()
