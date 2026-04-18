"""
=============================================================================
ShelfMind AI - Complete Kaggle Training Notebook (v2)
=============================================================================
Run this on Kaggle with GPU enabled (T4/P100).

SETUP on Kaggle:
1. Create a new Notebook → Settings → Accelerator → GPU T4 x2
2. Copy-paste this entire script and run all cells

NOTE: SKU-110K is auto-downloaded (13.6 GB streaming — no double disk usage)

TRAINING PIPELINE:
  Part 0: Download SKU-110K + convert to BOTH formats (YOLO + COCO)
  Part 1: RF-DETR-Small fine-tuning on SKU-110K  (~3-4 hours) ← FIRST
  Part 2: YOLO26s fine-tuning on SKU-110K         (~2.5 hours) ← SECOND
  Part 3: Extract 50K+ product crops for DINOv2 fine-tuning
  Part 4: Comparison table + Package all outputs

OUTPUT:
   - rfdetr_shelf_best.pt      (~60 MB)  -> RF-DETR detector
   - yolo_shelf_best.pt        (~20 MB)  -> YOLO26 detector
   - extracted_crops.zip        (~2 GB)  -> Crops for DINOv2 (Lightning.ai)
   - comparison_results.json              -> YOLO vs RF-DETR metrics
=============================================================================
"""

# %% [markdown]
# # ShelfMind AI - Training Pipeline v2
# ## Part 0: Setup & Install

# %%
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("ultralytics")
install("rfdetr")
install("faster-coco-eval")
install("pycocotools")
install("faiss-cpu")
install("joblib")

# %%
import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # SKU-110K has some truncated images
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
import gc

plt.style.use("dark_background")
COLORS = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94", "#c7ceea"]

import torch
import torch.nn as nn
from torchvision import transforms

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# === CONFIGURATION ===
KAGGLE_MODE = os.path.exists("/kaggle/input")
OUTPUT_DIR = "/kaggle/working/shelfmind_models" if KAGGLE_MODE else "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)
print(f"Mode: {'Kaggle' if KAGGLE_MODE else 'Local'}")
print(f"Output: {OUTPUT_DIR}")
print(f"Figures: {FIG_DIR}")


# %% [markdown]
# ---
# # PART 0: Download SKU-110K + Convert Annotations
# ---
# Download once, convert to BOTH formats so Part 1 and Part 2 can run immediately.

# %%
print("\n" + "="*60)
print("  PART 0: DOWNLOAD SKU-110K + CONVERT ANNOTATIONS")
print("="*60)

SKU_DIR = "/kaggle/working/datasets/SKU-110K"
TAR_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
os.makedirs("/kaggle/working/datasets", exist_ok=True)

if not os.path.exists(SKU_DIR):
    print("  Downloading + extracting SKU-110K (streaming, ~8 min)...")
    print("  (tar.gz streams directly to extraction — no double disk usage)")
    subprocess.run(
        f"curl -L {TAR_URL} | tar -xz -C /kaggle/working/datasets/",
        shell=True, check=True
    )
    extracted_dir = "/kaggle/working/datasets/SKU110K_fixed"
    if os.path.exists(extracted_dir):
        os.rename(extracted_dir, SKU_DIR)
    print(f"  [OK] SKU-110K extracted!")
    os.system("df -h /kaggle/working | tail -1")
else:
    print(f"  SKU-110K already exists at {SKU_DIR}")

# %%
# === Convert to COCO format (for RF-DETR) ===

def convert_sku110k_to_coco(sku_dir):
    """Convert SKU-110K CSV annotations to COCO JSON format for RF-DETR."""
    ann_dir = os.path.join(sku_dir, "annotations")
    if not os.path.exists(ann_dir):
        print(f"  [ERROR] Annotations directory not found: {ann_dir}")
        return None

    coco_base = os.path.join(sku_dir, "coco_format")
    os.makedirs(coco_base, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        csv_file = os.path.join(ann_dir, f"annotations_{split_name}.csv")
        if not os.path.exists(csv_file):
            print(f"  [SKIP] {csv_file} not found")
            continue

        # RF-DETR expects folder named "valid" not "val"
        folder_name = "valid" if split_name == "val" else split_name
        print(f"  Converting {split_name} to COCO format (folder: {folder_name})...")
        df = pd.read_csv(csv_file, header=None,
                         names=["image", "x1", "y1", "x2", "y2", "cls", "img_w", "img_h"])

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "object", "supercategory": "product"}]
        }

        image_id_map = {}
        ann_id = 0

        for img_name, group in df.groupby("image"):
            if img_name not in image_id_map:
                img_id = len(image_id_map)
                image_id_map[img_name] = img_id
                row0 = group.iloc[0]
                coco["images"].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": int(row0["img_w"]),
                    "height": int(row0["img_h"]),
                })

            for _, row in group.iterrows():
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w < 1 or h < 1:
                    continue
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id_map[img_name],
                    "category_id": 0,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        split_dir = os.path.join(coco_base, folder_name)
        os.makedirs(split_dir, exist_ok=True)

        src_images = os.path.join(sku_dir, "images")
        for img_info in coco["images"]:
            src = os.path.join(src_images, img_info["file_name"])
            dst = os.path.join(split_dir, img_info["file_name"])
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

        json_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(json_path, "w") as f:
            json.dump(coco, f)

        print(f"    {split_name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    print(f"  [OK] COCO format ready: {coco_base}")
    return coco_base


print("\n--- Converting to COCO format (for RF-DETR) ---")
coco_dataset_dir = convert_sku110k_to_coco(SKU_DIR)

# %%
# === Convert to YOLO format ===

def convert_sku110k_to_yolo(sku_dir):
    """Convert SKU-110K CSV annotations to YOLO text format."""
    labels_dir = os.path.join(sku_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    ann_dir = os.path.join(sku_dir, "annotations")
    if not os.path.exists(ann_dir):
        print(f"  [ERROR] Annotations directory not found: {ann_dir}")
        return None

    for split_name in ["train", "val", "test"]:
        csv_file = os.path.join(ann_dir, f"annotations_{split_name}.csv")
        if not os.path.exists(csv_file):
            continue

        print(f"  Converting {split_name} to YOLO format...")
        df = pd.read_csv(csv_file, header=None,
                         names=["image", "x1", "y1", "x2", "y2", "cls", "img_w", "img_h"])

        unique_images = df["image"].unique()
        split_file = os.path.join(sku_dir, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for img_name in unique_images:
                f.write(f"./images/{img_name}\n")

        for img_name, group in df.groupby("image"):
            label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(label_file, "w") as f:
                for _, row in group.iterrows():
                    w, h = row["img_w"], row["img_h"]
                    cx = max(0, min(1, ((row["x1"] + row["x2"]) / 2) / w))
                    cy = max(0, min(1, ((row["y1"] + row["y2"]) / 2) / h))
                    bw = max(0.001, min(1, abs(row["x2"] - row["x1"]) / w))
                    bh = max(0.001, min(1, abs(row["y2"] - row["y1"]) / h))
                    f.write(f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")

        print(f"    {split_name}: {len(unique_images)} images, {len(df)} bboxes")

    yaml_content = f"""path: {sku_dir}
train: train.txt
val: val.txt
test: test.txt

names:
  0: object

nc: 1
"""
    yaml_path = os.path.join(sku_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  [OK] YOLO YAML ready: {yaml_path}")
    return yaml_path


print("\n--- Converting to YOLO format (for YOLO26s) ---")
yolo_yaml_path = convert_sku110k_to_yolo(SKU_DIR)

print("\n[OK] Both formats ready! Proceeding to training...\n")


# %% [markdown]
# ---
# # PART 1: RF-DETR-Small Fine-Tuning on SKU-110K (FIRST)
# ---
# RF-DETR: Real-time transformer detector with DINOv2 backbone (ICLR 2026)
# Running this FIRST so we can catch any errors early.

# %%
print("\n" + "="*60)
print("  PART 1: RF-DETR-Small TRAINING ON SKU-110K  [FIRST]")
print("  (DINOv2 backbone, transformer attention, ICLR 2026)")
print("  Running first to catch any errors early!")
print("="*60)

from rfdetr import RFDETRSmall

print("\nStarting RF-DETR-Small training...")
print("  Backbone:        DINOv2 ViT (pretrained)")
print("  Attention:        Global self-attention (better for dense shelves)")
print("  batch_size:       2 (T4 15.6GB — dense SKU-110K images)")
print("  grad_accum:       8 (effective batch = 2×8 = 16)")
print("  Expected time:    ~2-3 hours on T4 GPU\n")

rfdetr_model = RFDETRSmall()

rfdetr_model.train(
    dataset_dir=coco_dataset_dir,
    epochs=30,
    batch_size=2,           # T4 OOMs at batch=4 due to multi-scale 840px
    grad_accum_steps=8,     # Effective batch size = 2 × 8 = 16
    lr=1e-4,
    output_dir=os.path.join(OUTPUT_DIR, "rfdetr_sku110k"),
)

print("\n[OK] RF-DETR-Small training complete!")

# %%
# === Save best RF-DETR model ===
rfdetr_output_dir = os.path.join(OUTPUT_DIR, "rfdetr_sku110k")
rfdetr_best_dst = os.path.join(OUTPUT_DIR, "rfdetr_shelf_best.pt")

rfdetr_ckpt = None
for candidate in [
    os.path.join(rfdetr_output_dir, "best_checkpoint.pth"),
    os.path.join(rfdetr_output_dir, "checkpoint_best.pth"),
    os.path.join(rfdetr_output_dir, "best.pth"),
    os.path.join(rfdetr_output_dir, "model_best.pth"),
]:
    if os.path.exists(candidate):
        rfdetr_ckpt = candidate
        break

if rfdetr_ckpt is None:
    for f in sorted(glob.glob(os.path.join(rfdetr_output_dir, "**", "*.pth"), recursive=True)):
        rfdetr_ckpt = f
        break

if rfdetr_ckpt:
    shutil.copy2(rfdetr_ckpt, rfdetr_best_dst)
    size_mb = os.path.getsize(rfdetr_best_dst) / 1e6
    print(f"\n[OK] RF-DETR model saved: {rfdetr_best_dst} ({size_mb:.1f} MB)")
else:
    print("[WARNING] RF-DETR checkpoint not found! Listing output directory:")
    for f in glob.glob(os.path.join(rfdetr_output_dir, "**", "*"), recursive=True):
        print(f"  {f}")

# %%
# === Validate RF-DETR ===
print("\nValidating RF-DETR-Small on SKU-110K val set...")

rfdetr_metrics = {"model": "RF-DETR-Small", "mAP50": 0, "mAP50_95": 0,
                  "precision": 0, "recall": 0, "size_MB": 0}

try:
    # NOTE: We renamed "val" → "valid" for RF-DETR training, so use "valid" here
    val_coco_dir = os.path.join(coco_dataset_dir, "valid")
    val_json = os.path.join(val_coco_dir, "_annotations.coco.json")

    if os.path.exists(val_json):
        with open(val_json) as f:
            val_coco = json.load(f)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import supervision as sv

        gt_coco = COCO(val_json)

        # Load fine-tuned RF-DETR using correct API: from_checkpoint()
        if rfdetr_ckpt:
            rfdetr_eval = RFDETRSmall.from_checkpoint(rfdetr_ckpt)
        else:
            rfdetr_eval = RFDETRSmall()

        predictions = []
        print("  Running RF-DETR predictions on val set...")
        for idx, img_info in enumerate(val_coco["images"]):
            img_path = os.path.join(val_coco_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            if idx % 100 == 0:
                print(f"    [{idx}/{len(val_coco['images'])}]...")
            try:
                # predict() returns supervision.Detections object
                detections = rfdetr_eval.predict(img_path, threshold=0.3)
                # Extract bbox and confidence from sv.Detections
                if detections and len(detections) > 0:
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        predictions.append({
                            "image_id": img_info["id"],
                            "category_id": 0,
                            "bbox": [float(x1), float(y1), w, h],  # COCO format
                            "score": float(detections.confidence[i]),
                        })
            except Exception:
                continue

        if predictions:
            pred_coco = gt_coco.loadRes(predictions)
            coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            rfdetr_metrics = {
                "model": "RF-DETR-Small",
                "mAP50": float(coco_eval.stats[1]),
                "mAP50_95": float(coco_eval.stats[0]),
                "precision": float(coco_eval.stats[8]) if len(coco_eval.stats) > 8 else 0.0,
                "recall": float(coco_eval.stats[6]) if len(coco_eval.stats) > 6 else 0.0,
                "size_MB": os.path.getsize(rfdetr_best_dst) / 1e6 if os.path.exists(rfdetr_best_dst) else 0,
            }

        del rfdetr_eval
except Exception as e:
    print(f"  [WARNING] RF-DETR evaluation error: {e}")
    print("  Will use training logs for metrics")

print(f"\n--- RF-DETR-Small Validation Results ---")
for k, v in rfdetr_metrics.items():
    print(f"  {k}: {v}")

# Free GPU
del rfdetr_model
torch.cuda.empty_cache()
gc.collect()
print("\n[OK] GPU memory freed for YOLO26s training")


# %% [markdown]
# ---
# # PART 2: YOLO26s Fine-Tuning on SKU-110K (SECOND)
# ---
# YOLO26s (2025) — NMS-free, 43% faster on CPU, STAL for small objects
# We already know this works (mAP50=0.895 from previous run)

# %%
from ultralytics import YOLO

print("\n" + "="*60)
print("  PART 2: YOLO26s TRAINING ON SKU-110K  [SECOND]")
print("  (CNN backbone, NMS-free, optimized for dense detection)")
print("="*60)

print("\nStarting YOLO26s training...")
model = YOLO("yolo26s.pt")

results = model.train(
    data=yolo_yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    patience=10,
    lr0=0.001,
    lrf=0.01,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    flipud=0.0,
    fliplr=0.5,
    degrees=3.0,
    scale=0.3,
    translate=0.1,
    project=OUTPUT_DIR,
    name="yolo_sku110k",
    exist_ok=True,
    verbose=True,
)

# %%
# === Save best YOLO model ===
best_src = os.path.join(OUTPUT_DIR, "yolo_sku110k", "weights", "best.pt")
yolo_best_dst = os.path.join(OUTPUT_DIR, "yolo_shelf_best.pt")

if os.path.exists(best_src):
    shutil.copy2(best_src, yolo_best_dst)
    size_mb = os.path.getsize(yolo_best_dst) / 1e6
    print(f"\n[OK] YOLO model saved: {yolo_best_dst} ({size_mb:.1f} MB)")
else:
    last_src = os.path.join(OUTPUT_DIR, "yolo_sku110k", "weights", "last.pt")
    shutil.copy2(last_src, yolo_best_dst)
    print(f"[WARNING] best.pt not found, using last.pt")

# %%
# === Validate YOLO ===
print("\nValidating YOLO26s on SKU-110K val set...")
val_model = YOLO(yolo_best_dst)
yolo_val_results = val_model.val(data=yolo_yaml_path, imgsz=640, batch=16)

yolo_metrics = {
    "model": "YOLO26s",
    "mAP50": float(yolo_val_results.box.map50),
    "mAP50_95": float(yolo_val_results.box.map),
    "precision": float(yolo_val_results.box.mp),
    "recall": float(yolo_val_results.box.mr),
    "params_M": 9.9,
    "size_MB": os.path.getsize(yolo_best_dst) / 1e6,
}

print(f"\n--- YOLO26s Validation Results ---")
for k, v in yolo_metrics.items():
    print(f"  {k}: {v}")

# %%
# === VIZ: YOLO Training Curves ===

def plot_yolo_training_curves(results_dir, save_dir):
    """Plot YOLO training loss, mAP, precision, recall curves."""
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] results.csv not found")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("ShelfMind AI - YOLO26s Training on SKU-110K", fontsize=18, color="white", fontweight="bold")

    for idx, (col, title, color) in enumerate([
        ("train/box_loss", "Box Loss", COLORS[0]),
        ("train/cls_loss", "Class Loss", COLORS[1]),
        ("train/dfl_loss", "DFL Loss", COLORS[2]),
    ]):
        ax = axes[0, idx]
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2, label="Train")
            val_col = col.replace("train/", "val/")
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], color=color, linewidth=2, linestyle="--", alpha=0.7, label="Val")
            ax.legend(fontsize=10)
        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.2)

    for idx, (col, title, color) in enumerate([
        ("metrics/precision(B)", "Precision", COLORS[3]),
        ("metrics/recall(B)", "Recall", COLORS[4]),
        ("metrics/mAP50(B)", "mAP@50", COLORS[5]),
    ]):
        ax = axes[1, idx]
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2.5)
            ax.fill_between(df["epoch"], 0, df[col], alpha=0.15, color=color)
            best_idx = df[col].idxmax()
            best_val, best_ep = df[col].iloc[best_idx], df["epoch"].iloc[best_idx]
            ax.scatter([best_ep], [best_val], color=color, s=100, zorder=5, edgecolors="white")
            ax.annotate(f"Best: {best_val:.3f}", xy=(best_ep, best_val),
                       xytext=(10, 10), textcoords="offset points", color=color, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Value"); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "yolo_training_curves.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Training curves saved")


plot_yolo_training_curves(os.path.join(OUTPUT_DIR, "yolo_sku110k"), FIG_DIR)

# Free GPU
del model, val_model
torch.cuda.empty_cache(); gc.collect()

# %%
# === VIZ: YOLO vs RF-DETR Comparison ===

def plot_model_comparison(yolo_m, rfdetr_m, save_dir):
    """Side-by-side comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("ShelfMind AI — YOLO26s vs RF-DETR-Small on SKU-110K",
                 fontsize=18, color="white", fontweight="bold")

    ax = axes[0]
    names = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    yv = [yolo_m["mAP50"], yolo_m["mAP50_95"], yolo_m["precision"], yolo_m["recall"]]
    rv = [rfdetr_m["mAP50"], rfdetr_m["mAP50_95"], rfdetr_m["precision"], rfdetr_m["recall"]]

    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, yv, w, label="YOLO26s", color=COLORS[0], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, rv, w, label="RF-DETR-Small", color=COLORS[1], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.legend(fontsize=12)
    ax.set_title("Detection Metrics", fontsize=14, color="white"); ax.grid(True, axis="y", alpha=0.2)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="white")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, color="white")

    ax2 = axes[1]
    ax2.axis("off")
    tdata = [
        ["Feature", "YOLO26s", "RF-DETR-Small"],
        ["Architecture", "CNN (CSPNet)", "Transformer (DINOv2)"],
        ["Attention", "Local (conv)", "Global (self-attn)"],
        ["NMS", "NMS-free ✓", "NMS-free ✓"],
        ["Backbone", "CSPDarknet", "DINOv2 ViT"],
        ["Model Size", f"{yolo_m.get('size_MB', 20):.0f} MB", f"{rfdetr_m.get('size_MB', 100):.0f} MB"],
        ["mAP@50", f"{yolo_m['mAP50']:.4f}", f"{rfdetr_m['mAP50']:.4f}"],
        ["Best For", "CPU/Edge", "GPU/Accuracy"],
    ]
    table = ax2.table(cellText=tdata, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 1.8)
    for j in range(3):
        table[0, j].set_facecolor("#00d4aa")
        table[0, j].set_text_props(color="black", fontweight="bold")
    for i in range(1, len(tdata)):
        for j in range(3):
            table[i, j].set_facecolor("#1a1a2e")
            table[i, j].set_text_props(color="white")
            table[i, j].set_edgecolor("#333")
    ax2.set_title("Architecture Comparison", fontsize=14, color="white", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(save_dir, "yolo_vs_rfdetr_comparison.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Comparison chart saved")


plot_model_comparison(yolo_metrics, rfdetr_metrics, FIG_DIR)

comparison = {"yolo26s": yolo_metrics, "rfdetr_base": rfdetr_metrics}
with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
    json.dump(comparison, f, indent=2)


# %% [markdown]
# ---
# # PART 3: Extract Product Crops for DINOv2 Fine-Tuning
# ---
# Use trained YOLO26s (faster on CPU) to extract 50K+ product crops from ALL
# 11,743 SKU-110K images. Saved as zip for Lightning.ai upload.

# %%
print("\n" + "="*60)
print("  PART 3: EXTRACTING PRODUCT CROPS FOR DINOv2 FINE-TUNING")
print("  (Using fine-tuned YOLO26s on ALL 11,743 images)")
print("="*60)

def find_all_sku_images(sku_dir):
    """Find all SKU-110K images."""
    all_imgs = []
    images_dir = os.path.join(sku_dir, "images")
    if os.path.exists(images_dir):
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            all_imgs.extend(glob.glob(os.path.join(images_dir, ext)))
    print(f"  Found {len(all_imgs)} total images")
    return all_imgs


def extract_all_crops(yolo_model_path, image_files, output_dir, max_crops_per_image=20):
    """Extract product crops from ALL images for DINOv2 fine-tuning."""
    det_model = YOLO(yolo_model_path)

    crops_dir = os.path.join(output_dir, "crops_for_dinov2")
    os.makedirs(crops_dir, exist_ok=True)

    metadata = []
    crop_count = 0

    print(f"  Extracting crops from {len(image_files)} images (max {max_crops_per_image}/image)...")

    for img_idx, img_path in enumerate(image_files):
        if img_idx % 500 == 0:
            print(f"    [{img_idx}/{len(image_files)}] {crop_count} crops so far...")

        try:
            results = det_model(img_path, conf=0.35, verbose=False)
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        source_name = os.path.basename(img_path)

        for det_idx, box in enumerate(results[0].boxes):
            if det_idx >= max_crops_per_image:
                break

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            if conf < 0.35:
                continue

            pad = 3
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)

            if (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_filename = f"crop_{crop_count:06d}.jpg"
            crop.save(os.path.join(crops_dir, crop_filename), "JPEG", quality=85)

            metadata.append({
                "crop_id": crop_filename,
                "source_image": source_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "width": x2 - x1,
                "height": y2 - y1,
            })
            crop_count += 1

    meta_path = os.path.join(output_dir, "crop_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    print(f"\n  [OK] Extracted {crop_count} product crops")
    print(f"  [OK] Crops dir: {crops_dir}")
    print(f"  [OK] Metadata: {meta_path}")
    return crops_dir, metadata


all_images = find_all_sku_images(SKU_DIR)
crops_dir, crop_metadata = extract_all_crops(yolo_best_dst, all_images, OUTPUT_DIR, max_crops_per_image=20)

# %%
# === Create ZIP for Lightning.ai ===

import zipfile

print("\nCreating crops zip for Lightning.ai upload...")
crops_zip_path = os.path.join(OUTPUT_DIR, "extracted_crops.zip")

with zipfile.ZipFile(crops_zip_path, "w", zipfile.ZIP_STORED) as zf:
    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*.jpg")))
    for i, crop_file in enumerate(crop_files):
        zf.write(crop_file, f"crops/{os.path.basename(crop_file)}")
        if i % 10000 == 0 and i > 0:
            print(f"    Zipped {i}/{len(crop_files)} crops...")

    meta_path = os.path.join(OUTPUT_DIR, "crop_metadata.json")
    if os.path.exists(meta_path):
        zf.write(meta_path, "crop_metadata.json")

zip_size = os.path.getsize(crops_zip_path) / 1e6
print(f"\n[OK] Crops zip: {crops_zip_path} ({zip_size:.1f} MB, {len(crop_files)} crops)")

# %%
# === VIZ 3: Crop grid + statistics ===

def plot_crop_grid(c_dir, save_dir, n_show=40):
    c_files = sorted(glob.glob(os.path.join(c_dir, "*.jpg")))
    if not c_files: return
    np.random.seed(42)
    samples = np.random.choice(c_files, min(n_show, len(c_files)), replace=False)
    cols = 10; rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.2 * rows))
    fig.suptitle(f"Product Crops for DINOv2 ({len(c_files)} total)", fontsize=16, color="white", fontweight="bold")
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        if i < len(samples):
            try: ax.imshow(Image.open(samples[i]))
            except: pass
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(save_dir, "dinov2_crop_grid.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Crop grid saved")

def plot_crop_stats(meta, save_dir):
    if not meta: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Crop Extraction Statistics", fontsize=16, color="white", fontweight="bold")
    ws = [m["width"] for m in meta]
    axes[0].hist(ws, bins=50, color=COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[0].set_title(f"Widths (median: {np.median(ws):.0f}px)", color="white"); axes[0].grid(True, alpha=0.2)
    hs = [m["height"] for m in meta]
    axes[1].hist(hs, bins=50, color=COLORS[2], alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].set_title(f"Heights (median: {np.median(hs):.0f}px)", color="white"); axes[1].grid(True, alpha=0.2)
    cs = [m["confidence"] for m in meta]
    axes[2].hist(cs, bins=50, color=COLORS[3], alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[2].set_title(f"Confidence (mean: {np.mean(cs):.3f})", color="white"); axes[2].grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(save_dir, "crop_statistics.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Crop statistics saved")

plot_crop_grid(crops_dir, FIG_DIR)
plot_crop_stats(crop_metadata, FIG_DIR)


# %% [markdown]
# ---
# # PART 4: Package All Models & Outputs
# ---

# %%
print("\n" + "="*60)
print("   SHELFMIND AI - ALL TRAINING COMPLETE!")
print("="*60)

model_files = [
    ("rfdetr_shelf_best.pt", "Product Detection — RF-DETR-Small (DINOv2 transformer, SKU-110K)"),
    ("yolo_shelf_best.pt", "Product Detection — YOLO26s (CNN, NMS-free, SKU-110K)"),
    ("comparison_results.json", "YOLO vs RF-DETR comparison metrics"),
    ("crop_metadata.json", "Metadata for extracted product crops"),
]

print(f"\nOutput directory: {OUTPUT_DIR}\n")
total_size = 0
for filename, description in model_files:
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1e6
        total_size += size
        print(f"  [OK]      {filename:35s} {size:6.1f} MB  -  {description}")
    else:
        print(f"  [MISSING] {filename:35s}          -  {description}")

if os.path.exists(crops_zip_path):
    cz = os.path.getsize(crops_zip_path) / 1e6
    n_crops = len(glob.glob(os.path.join(crops_dir, "*.jpg")))
    print(f"  [OK]      {'extracted_crops.zip':35s} {cz:6.1f} MB  -  {n_crops} crops for DINOv2")

# === Separate ZIP for RF-DETR ===
rfdetr_zip = os.path.join(OUTPUT_DIR, "rfdetr_shelf.zip")
with zipfile.ZipFile(rfdetr_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(os.path.join(OUTPUT_DIR, "rfdetr_shelf_best.pt")):
        zf.write(os.path.join(OUTPUT_DIR, "rfdetr_shelf_best.pt"), "rfdetr_shelf_best.pt")
    if os.path.exists(os.path.join(OUTPUT_DIR, "comparison_results.json")):
        zf.write(os.path.join(OUTPUT_DIR, "comparison_results.json"), "comparison_results.json")
    if os.path.exists(FIG_DIR):
        for fig_file in glob.glob(os.path.join(FIG_DIR, "*.png")):
            zf.write(fig_file, f"visualizations/{os.path.basename(fig_file)}")
rz = os.path.getsize(rfdetr_zip) / 1e6
print(f"\n  [OK] rfdetr_shelf.zip        ({rz:.1f} MB)")

# === Separate ZIP for YOLO ===
yolo_zip = os.path.join(OUTPUT_DIR, "yolo26_shelf.zip")
with zipfile.ZipFile(yolo_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(os.path.join(OUTPUT_DIR, "yolo_shelf_best.pt")):
        zf.write(os.path.join(OUTPUT_DIR, "yolo_shelf_best.pt"), "yolo_shelf_best.pt")
    if os.path.exists(os.path.join(OUTPUT_DIR, "comparison_results.json")):
        zf.write(os.path.join(OUTPUT_DIR, "comparison_results.json"), "comparison_results.json")
    # Include YOLO training curves
    yolo_results_csv = os.path.join(OUTPUT_DIR, "yolo_sku110k", "results.csv")
    if os.path.exists(yolo_results_csv):
        zf.write(yolo_results_csv, "results.csv")
    if os.path.exists(FIG_DIR):
        for fig_file in glob.glob(os.path.join(FIG_DIR, "*.png")):
            zf.write(fig_file, f"visualizations/{os.path.basename(fig_file)}")
yz = os.path.getsize(yolo_zip) / 1e6
print(f"  [OK] yolo26_shelf.zip        ({yz:.1f} MB)")

# crops zip already created above
print(f"  [OK] extracted_crops.zip     ({zip_size:.1f} MB)")

# Print final comparison
print("\n" + "="*60)
print("  YOLO26s vs RF-DETR-Small — FINAL COMPARISON")
print("="*60)
print(f"  {'Metric':<20s} {'YOLO26s':>12s} {'RF-DETR-Small':>16s}")
print(f"  {'-'*47}")
print(f"  {'mAP@50':<20s} {yolo_metrics['mAP50']:>12.4f} {rfdetr_metrics['mAP50']:>15.4f}")
print(f"  {'mAP@50-95':<20s} {yolo_metrics['mAP50_95']:>12.4f} {rfdetr_metrics['mAP50_95']:>15.4f}")
print(f"  {'Precision':<20s} {yolo_metrics['precision']:>12.4f} {rfdetr_metrics['precision']:>15.4f}")
print(f"  {'Recall':<20s} {yolo_metrics['recall']:>12.4f} {rfdetr_metrics['recall']:>15.4f}")
print(f"  {'Model Size':<20s} {yolo_metrics.get('size_MB', 20):>11.1f}M {rfdetr_metrics.get('size_MB', 100):>14.1f}M")

print(f"\n{'='*60}")
print("  DOWNLOADABLE FILES (3 separate zips):")
print(f"  1. rfdetr_shelf.zip          ({rz:.1f} MB)  — RF-DETR weights + visualizations")
print(f"  2. yolo26_shelf.zip          ({yz:.1f} MB)  — YOLO26 weights + visualizations")
print(f"  3. extracted_crops.zip       ({zip_size:.1f} MB)  — Crops for DINOv2 (Lightning.ai)")
print(f"\n  NEXT STEPS:")
print("  1. Download all 3 zip files from Kaggle Output tab")
print("  2. Upload extracted_crops.zip to Lightning.ai")
print("  3. Run dinov2_finetune.py (15 credits)")
print("  4. Integrate all models into web app")
print(f"{'='*60}")
