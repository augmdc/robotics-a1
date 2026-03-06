"""
graphs.py  —  Task 2 Reporting Artifacts
Course: COSC-4366EL-01 Autonomous Mobile Robotics  |  Assignment 1 (2026 W)

Generates three figures saved to ./reports/:
  1. class_table.png     — per-class Precision / Recall / AP@0.5
  2. confusion_matrix.png — heatmap highlighting Left/Right/Forward mix-ups
  3. timing_table.png    — per-frame inference time and FPS benchmark

Usage:
  python graphs.py
  python graphs.py --model best_yolo.pt --data fira-signs.v4i.yolov8/data.yaml
"""

from __future__ import annotations

import argparse
import time
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES   = ["Dead End", "No Entry", "Proceed Forward",
                 "Proceed Left", "Proceed Right", "Stop"]
SHORT_NAMES   = ["Dead\nEnd", "No\nEntry", "Fwd", "Left", "Right", "Stop"]
DEFAULT_MODEL = "best_yolo.pt"
DEFAULT_DATA  = "fira-signs.v4i.yolov8/data.yaml"
OUT_DIR       = Path("reports")

HDR_COLOR  = "#2c5282"   # dark blue header
WEAK_COLOR = "#fed7d7"   # light red for values below threshold


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _run_val(model_path: str, data_yaml: str):
    from ultralytics import YOLO  # type: ignore[import]
    model   = YOLO(model_path)
    metrics = model.val(
        data=data_yaml, split="test", imgsz=640,
        verbose=False, plots=False, save=False,
    )
    return metrics


# ---------------------------------------------------------------------------
# 1. Per-class detection table
# ---------------------------------------------------------------------------

def save_class_table(metrics, out_dir: Path) -> None:
    p   = metrics.box.p     # precision per class
    r   = metrics.box.r     # recall per class
    ap  = metrics.box.ap50  # AP@0.5 per class

    col_labels = ["Class", "Precision", "Recall", "AP@0.5"]
    rows = [
        [CLASS_NAMES[i], f"{p[i]:.3f}", f"{r[i]:.3f}", f"{ap[i]:.3f}"]
        for i in range(len(CLASS_NAMES))
    ]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(HDR_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight weak Recall (<0.65) and Precision (<0.75)
    for i, row in enumerate(rows):
        if float(row[2]) < 0.65:
            tbl[i + 1, 2].set_facecolor(WEAK_COLOR)
        if float(row[1]) < 0.75:
            tbl[i + 1, 1].set_facecolor(WEAK_COLOR)

    ax.set_title("Per-Class Detection Metrics  (test split)",
                 fontsize=13, fontweight="bold", pad=14)

    path = out_dir / "class_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[graphs] Saved → {path}")


# ---------------------------------------------------------------------------
# 2. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def save_confusion_matrix(metrics, out_dir: Path) -> None:
    nc   = len(CLASS_NAMES)
    raw  = metrics.confusion_matrix.matrix  # shape (nc+1, nc+1)
    data = raw[:nc, :nc].astype(int)        # drop background row/col

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues")

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(SHORT_NAMES, fontsize=9)
    ax.set_yticklabels(SHORT_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix  (test split)", fontsize=13, fontweight="bold")

    thresh = data.max() / 2
    for i in range(nc):
        for j in range(nc):
            ax.text(j, i, str(data[i, j]),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if data[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[graphs] Saved → {path}")


# ---------------------------------------------------------------------------
# 3. Inference timing table
# ---------------------------------------------------------------------------

def save_timing_table(model_path: str, test_images_dir: str,
                      out_dir: Path, n_warmup: int = 5) -> None:
    from ultralytics import YOLO  # type: ignore[import]

    images = (sorted(glob.glob(f"{test_images_dir}/*.jpg")) +
              sorted(glob.glob(f"{test_images_dir}/*.png")))
    if not images:
        print(f"[graphs] No images found in {test_images_dir} — skipping timing")
        return

    model = YOLO(model_path)

    # Warm-up passes (excluded from stats)
    for img in images[:n_warmup]:
        model.predict(source=img, imgsz=640, verbose=False)

    times_ms: list[float] = []
    for img in images:
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=640, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000)

    t = np.array(times_ms)
    rows = [
        ["Images timed",        str(len(t))],
        ["Avg inference time",  f"{t.mean():.1f} ms"],
        ["Min / Max",           f"{t.min():.1f} / {t.max():.1f} ms"],
        ["Std dev",             f"{t.std():.1f} ms"],
        ["Avg FPS",             f"{1000 / t.mean():.1f}"],
        ["P95 latency",         f"{np.percentile(t, 95):.1f} ms"],
    ]

    fig, ax = plt.subplots(figsize=(5, 3.0))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=["Metric", "Value"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for j in range(2):
        tbl[0, j].set_facecolor(HDR_COLOR)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Inference Timing Summary", fontsize=13, fontweight="bold", pad=14)

    path = out_dir / "timing_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[graphs] Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Task 2 reporting graphs")
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--data",        default=DEFAULT_DATA)
    ap.add_argument("--test-images", default="fira-signs.v4i.yolov8/test/images")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("[graphs] Running validation on test split...")
    metrics = _run_val(args.model, args.data)

    save_class_table(metrics, OUT_DIR)
    save_confusion_matrix(metrics, OUT_DIR)
    save_timing_table(args.model, args.test_images, OUT_DIR)

    print(f"\n[graphs] All artifacts saved → ./{OUT_DIR}/")
