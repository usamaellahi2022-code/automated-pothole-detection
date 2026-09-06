"""
=============================================================
  Automated Pothole Detection System - BS Project
  IQRA National University, Peshawar
=============================================================
  Models: YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s
  All via: pip install ultralytics  (no repo cloning)
=============================================================
"""

import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIGURATION  ← edit only this block
# ──────────────────────────────────────────────
DATA_YAML   = r"D:\Saqib\work\pothole\pothole_dataset\data.yaml"
PROJECT_DIR = r"D:\Saqib\work\pothole_results"
EPOCHS      = 50        # increase to 100 for final submission
IMG_SIZE    = 640
BATCH_SIZE  = 16        # reduce to 8 if GPU runs out of memory
DEVICE      = "0"       # "0" = GPU,  "cpu" = CPU

# ──────────────────────────────────────────────
# MODELS  (all supported by ultralytics)
# ──────────────────────────────────────────────
MODELS = [
    # ("YOLOv8n",  "yolov8n.pt"),
    # ("YOLOv8s",  "yolov8s.pt"),
    # ("YOLOv11n", "yolo11n.pt"),
    # ("YOLOv11s", "yolo11s.pt"),
    ("YOLOv8m",  "yolov8m.pt"),
    ("YOLOv11m", "yolo11m.pt"),
]

# ──────────────────────────────────────────────

def banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def train_model(name, weights):
    from ultralytics import YOLO
    banner(f"Training {name}  [{weights}]")
    model = YOLO(weights)
    start = time.time()
    model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        device   = DEVICE,
        project  = PROJECT_DIR,
        name     = name,
        exist_ok = True,
        verbose  = True,
    )
    elapsed = time.time() - start
    print(f"\n  [{name}] finished in {elapsed/60:.1f} min")
    return elapsed


def parse_results(name, elapsed):
    record = {
        "Model":         name,
        "mAP50":         None,
        "mAP50-95":      None,
        "Precision":     None,
        "Recall":        None,
        "Train_Min":     round(elapsed / 60, 1),
    }
    csv_path = Path(PROJECT_DIR) / name / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        mapping = {
            "metrics/mAP50(B)":     "mAP50",
            "metrics/mAP50-95(B)":  "mAP50-95",
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)":    "Recall",
        }
        for src, dst in mapping.items():
            if src in df.columns:
                record[dst] = round(float(df[src].max()), 4)
    return record


def save_comparison(records):
    df = pd.DataFrame(records).sort_values("mAP50", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    # CSV
    csv_out = Path(PROJECT_DIR) / "model_comparison.csv"
    df.to_csv(csv_out, index=False)

    # Print table
    print("\n" + "="*60)
    print("  FINAL COMPARISON")
    print("="*60)
    print(df.to_string(index=False))

    # Bar chart
    metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Pothole Detection – Model Comparison\nIQRA National University, Peshawar",
                 fontsize=13, fontweight="bold")
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for ax, metric, color in zip(axes, metrics, colors):
        vals = df[metric].fillna(0).values
        bars = ax.bar(df["Model"], vals, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_out = Path(PROJECT_DIR) / "model_comparison.png"
    plt.savefig(str(chart_out), dpi=150, bbox_inches="tight")
    plt.close()

    # Training time chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(df["Model"], df["Train_Min"], color=colors[:len(df)], edgecolor="black")
    ax2.set_xlabel("Training Time (minutes)")
    ax2.set_title("Training Time per Model", fontweight="bold")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    for i, v in enumerate(df["Train_Min"]):
        ax2.text(v + 0.3, i, f"{v} min", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(str(Path(PROJECT_DIR) / "training_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # JSON report
    best = df.iloc[0]
    report = {
        "generated_at": datetime.now().isoformat(),
        "project":      "Automated Pothole Detection System",
        "university":   "IQRA National University, Peshawar",
        "best_model":   best["Model"],
        "best_mAP50":   best["mAP50"],
        "all_results":  records,
    }
    with open(Path(PROJECT_DIR) / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Best Model : {best['Model']}  (mAP50 = {best['mAP50']})")
    print(f"  Comparison : {csv_out}")
    print(f"  Chart      : {chart_out}")
    return df


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    records = []

    for name, weights in MODELS:
        try:
            elapsed = train_model(name, weights)
            record  = parse_results(name, elapsed)
        except Exception as e:
            print(f"\n  [ERROR] {name} failed: {e}")
            record = {"Model": name, "mAP50": 0, "mAP50-95": 0,
                      "Precision": 0, "Recall": 0, "Train_Min": 0}
        records.append(record)
        print(f"  → {name}: mAP50={record['mAP50']}")

    save_comparison(records)
    banner("ALL DONE")