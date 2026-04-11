"""
=============================================================================
ShelfMind AI - Complete Kaggle Training Notebook
=============================================================================
Run this on Kaggle with GPU enabled (T4/P100).

SETUP on Kaggle:
1. Create a new Notebook → Settings → Accelerator → GPU T4 x2
2. Add dataset: "m5-forecasting-accuracy" (from the M5 competition)
3. Copy-paste this entire script and run all cells

NOTE: SKU-110K is auto-downloaded by Ultralytics (13.6 GB) — no manual upload!
      Supermarket Shelves is optional (upload if you have it).

OUTPUT: 4 model files to download
   - yolo_shelf_best.pt       (~12 MB) -> Product detection  
   - sku_faiss_index.bin       (~2 MB)  -> SKU matching index
   - sku_metadata.json         (~1 MB)  -> SKU metadata
   - lgbm_forecast_model.pkl   (~5 MB)  -> Demand forecasting
=============================================================================
"""

# %% [markdown]
# # ShelfMind AI - Training Pipeline
# ## Part 0: Setup & Install

# %%
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("ultralytics")
install("faiss-cpu")
install("lightgbm")
install("joblib")
install("polars")  # Needed for SKU-110K label conversion

# %%
import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# Plot styling
plt.style.use("dark_background")
COLORS = ["#00d4aa", "#ff6b6b", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94", "#c7ceea"]
FIG_DIR = None  # Set after OUTPUT_DIR is defined

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

if KAGGLE_MODE:
    M5_DATA_DIR = "/kaggle/input/competitions/m5-forecasting-accuracy"
    OUTPUT_DIR = "/kaggle/working/shelfmind_models"
    # Optional: Supermarket shelves dataset (add via Kaggle datasets)
    SHELF_IMAGES_DIR = "/kaggle/input/supermarket-shelves-dataset"
else:
    M5_DATA_DIR = "data/m5-forecasting-accuracy"
    OUTPUT_DIR = "models"
    SHELF_IMAGES_DIR = "data/shelf_images/Supermarket shelves/Supermarket shelves"

os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)
print(f"Mode: {'Kaggle' if KAGGLE_MODE else 'Local'}")
print(f"Output: {OUTPUT_DIR}")
print(f"Figures: {FIG_DIR}")


# %% [markdown]
# ---
# # PART 1: YOLO Fine-Tuning on SKU-110K (11,762 images, 1M+ bboxes)
# ---
# 
# **Manual download + cleanup to fit in Kaggle's 19.5 GB disk.**
# Steps: Download tar.gz → Extract → DELETE tar.gz (frees 11.4 GB) → Convert labels → Train
#
# Dataset: 8,219 train + 588 val + 2,936 test images
# Classes: 1 (object/product on shelf)
# Source: CVPR 2019 paper "Precise Detection in Densely Packed Scenes"
# Model: YOLO26s (Jan 2026) — NMS-free, 43% faster on CPU, STAL for small objects

# %%
from ultralytics import YOLO
import subprocess
import tarfile

print("\n" + "="*60)
print("  PART 1: YOLO TRAINING ON SKU-110K")
print("  (Manual download + disk cleanup for Kaggle)")
print("="*60)

# === Step 1: Download SKU-110K ===
SKU_DIR = "/kaggle/working/datasets/SKU-110K"
TAR_PATH = "/kaggle/working/datasets/SKU110K_fixed.tar.gz"
TAR_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"

os.makedirs("/kaggle/working/datasets", exist_ok=True)

if not os.path.exists(SKU_DIR):
    # Stream download directly to tar — tar.gz NEVER saves to disk!
    # Only extracted files (~8 GB) touch disk. Fits in Kaggle's 19.5 GB.
    print("  Downloading + extracting SKU-110K (streaming, ~8 min)...")
    print("  (tar.gz streams directly to extraction — no double disk usage)")
    subprocess.run(
        f"curl -L {TAR_URL} | tar -xz -C /kaggle/working/datasets/",
        shell=True, check=True
    )
    
    # Rename extracted folder
    extracted_dir = "/kaggle/working/datasets/SKU110K_fixed"
    if os.path.exists(extracted_dir):
        os.rename(extracted_dir, SKU_DIR)
    
    print(f"  [OK] SKU-110K extracted!")
    os.system("df -h /kaggle/working | tail -1")
else:
    print(f"  SKU-110K already exists at {SKU_DIR}")

# %%
# === Step 2: Convert CSV annotations to YOLO format ===

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
            print(f"  [SKIP] {csv_file} not found")
            continue
        
        print(f"  Converting {split_name} annotations...")
        
        # Read CSV: image_name, x1, y1, x2, y2, class, image_width, image_height
        df = pd.read_csv(csv_file, header=None,
                         names=["image", "x1", "y1", "x2", "y2", "cls", "img_w", "img_h"])
        
        # Create split file listing images
        unique_images = df["image"].unique()
        split_file = os.path.join(sku_dir, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for img_name in unique_images:
                f.write(f"./images/{img_name}\n")
        
        # Convert each image's annotations to YOLO format
        for img_name, group in df.groupby("image"):
            label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            with open(label_file, "w") as f:
                for _, row in group.iterrows():
                    w, h = row["img_w"], row["img_h"]
                    # Convert xyxy to xywh normalized
                    cx = ((row["x1"] + row["x2"]) / 2) / w
                    cy = ((row["y1"] + row["y2"]) / 2) / h
                    bw = abs(row["x2"] - row["x1"]) / w
                    bh = abs(row["y2"] - row["y1"]) / h
                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0.001, min(1, bw))
                    bh = max(0.001, min(1, bh))
                    f.write(f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n")
        
        print(f"    {split_name}: {len(unique_images)} images, {len(df)} bboxes")
    
    # Create dataset YAML
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
    
    print(f"  [OK] YAML saved: {yaml_path}")
    return yaml_path


yolo_yaml_path = convert_sku110k_to_yolo(SKU_DIR)

# %%
# === Step 3: Train YOLO on SKU-110K ===

print("\nStarting YOLO26s training (NMS-free, optimized for dense shelf detection)...")
model = YOLO("yolo26s.pt")

results = model.train(
    data=yolo_yaml_path,     # Our manually created YAML
    epochs=30,               # 30 epochs (good for hackathon + fine-tuning)
    imgsz=640,               # Standard YOLO input size
    batch=16,                # T4 16GB can handle batch=16
    patience=10,             # Early stopping after 10 epochs no improvement
    lr0=0.001,               # Initial learning rate
    lrf=0.01,                # Final LR = lr0 * lrf
    augment=True,            # Data augmentation
    mosaic=1.0,              # Mosaic augmentation (great for dense packing)
    mixup=0.1,               # Mixup augmentation
    flipud=0.0,              # No vertical flip (shelves are always upright)
    fliplr=0.5,              # Horizontal flip
    degrees=3.0,             # Slight rotation (camera angle variation)
    scale=0.3,               # Scale augmentation
    translate=0.1,           # Translation
    project=OUTPUT_DIR,
    name="yolo_sku110k",
    exist_ok=True,
    verbose=True,
)

# %%
# === Save best model ===
best_src = os.path.join(OUTPUT_DIR, "yolo_sku110k", "weights", "best.pt")
best_dst = os.path.join(OUTPUT_DIR, "yolo_shelf_best.pt")

if os.path.exists(best_src):
    shutil.copy2(best_src, best_dst)
    size_mb = os.path.getsize(best_dst) / 1e6
    print(f"\n[OK] YOLO model saved: {best_dst} ({size_mb:.1f} MB)")
else:
    # Fallback to last.pt
    last_src = os.path.join(OUTPUT_DIR, "yolo_sku110k", "weights", "last.pt")
    shutil.copy2(last_src, best_dst)
    print(f"[WARNING] best.pt not found, using last.pt")

# %%
# === Validate ===
print("\nValidating on SKU-110K val set...")
val_model = YOLO(best_dst)
val_results = val_model.val(data=yolo_yaml_path, imgsz=640, batch=16)

print(f"\n--- SKU-110K Validation Results ---")
print(f"  mAP50:     {val_results.box.map50:.4f}")
print(f"  mAP50-95:  {val_results.box.map:.4f}")
print(f"  Precision:  {val_results.box.mp:.4f}")
print(f"  Recall:     {val_results.box.mr:.4f}")

# %% [markdown]
# ## VISUALIZATION 1: YOLO Training Curves

# %%
# === VIZ 1A: Training Curves from results.csv ===

def plot_yolo_training_curves(results_dir, save_dir):
    """Plot YOLO training loss, mAP, precision, recall curves."""
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] results.csv not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("ShelfMind AI - YOLO Training on SKU-110K", fontsize=18, color="white", fontweight="bold")
    
    # Row 1: Losses
    loss_cols = [
        ("train/box_loss", "Box Loss", COLORS[0]),
        ("train/cls_loss", "Class Loss", COLORS[1]),
        ("train/dfl_loss", "DFL Loss", COLORS[2]),
    ]
    for idx, (col, title, color) in enumerate(loss_cols):
        ax = axes[0, idx]
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2, label="Train")
            # Plot val loss if available
            val_col = col.replace("train/", "val/")
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], color=color, linewidth=2, linestyle="--", alpha=0.7, label="Val")
            ax.legend(fontsize=10)
        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.grid(True, alpha=0.2)
    
    # Row 2: Metrics
    metric_cols = [
        ("metrics/precision(B)", "Precision", COLORS[3]),
        ("metrics/recall(B)", "Recall", COLORS[4]),
        ("metrics/mAP50(B)", "mAP@50", COLORS[5]),
    ]
    for idx, (col, title, color) in enumerate(metric_cols):
        ax = axes[1, idx]
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2.5)
            ax.fill_between(df["epoch"], 0, df[col], alpha=0.15, color=color)
            # Mark best value
            best_idx = df[col].idxmax()
            best_val = df[col].iloc[best_idx]
            best_epoch = df["epoch"].iloc[best_idx]
            ax.scatter([best_epoch], [best_val], color=color, s=100, zorder=5, edgecolors="white")
            ax.annotate(f"Best: {best_val:.3f}", xy=(best_epoch, best_val),
                       xytext=(10, 10), textcoords="offset points", color=color, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=14, color="white")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, "yolo_training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Training curves saved: {save_path}")


yolo_results_dir = os.path.join(OUTPUT_DIR, "yolo_sku110k")
plot_yolo_training_curves(yolo_results_dir, FIG_DIR)

# %%
# === VIZ 1B: Sample Detection Results on Val Images ===

def plot_detection_samples(model_path, images_dir, save_dir, n_samples=6):
    """Run detection on sample images and visualize with bounding boxes."""
    det_model = YOLO(model_path)
    
    # Find images
    img_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        img_files.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    
    if not img_files:
        print(f"  [SKIP] No images found in {images_dir}")
        return
    
    # Random sample
    np.random.seed(42)
    sample_files = np.random.choice(img_files, min(n_samples, len(img_files)), replace=False)
    
    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 7 * rows))
    fig.suptitle("ShelfMind AI - Product Detection Results (SKU-110K)", fontsize=18, color="white", fontweight="bold")
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    box_colors = ["#00d4aa", "#ff6b6b", "#ffe66d", "#4ecdc4", "#a8e6cf"]
    
    for idx, img_path in enumerate(sample_files):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # Run detection
        results = det_model(img_path, conf=0.25, verbose=False)
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        
        # Draw bounding boxes
        n_detections = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            color = box_colors[n_detections % len(box_colors)]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.8)
            ax.add_patch(rect)
            n_detections += 1
        
        ax.set_title(f"{os.path.basename(img_path)}\n{n_detections} products detected",
                    fontsize=11, color="white")
        ax.axis("off")
    
    # Hide empty axes
    for idx in range(len(sample_files), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, "yolo_detection_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Detection samples saved: {save_path}")


# Find images from the downloaded SKU-110K
print("\nGenerating detection visualizations...")
for search_dir in [
    os.path.expanduser("~/datasets/SKU-110K/images"),
    "/kaggle/working/datasets/SKU-110K/images",
    "datasets/SKU-110K/images",
]:
    if os.path.exists(search_dir):
        plot_detection_samples(best_dst, search_dir, FIG_DIR, n_samples=6)
        break
else:
    print("  [SKIP] SKU-110K images not found for visualization")

# %%
# === VIZ 1C: Validation Metrics Summary Card  ===

def plot_metrics_card(map50, map50_95, precision, recall, save_dir):
    """Create a clean metrics summary card."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis("off")
    
    metrics = [
        ("mAP@50", map50, COLORS[0]),
        ("mAP@50-95", map50_95, COLORS[1]),
        ("Precision", precision, COLORS[2]),
        ("Recall", recall, COLORS[3]),
    ]
    
    for i, (name, value, color) in enumerate(metrics):
        x = 0.1 + i * 0.22
        # Circle background
        circle = plt.Circle((x + 0.08, 0.55), 0.12, color=color, alpha=0.15, transform=ax.transAxes)
        ax.add_patch(circle)
        # Value
        ax.text(x + 0.08, 0.6, f"{value:.3f}", transform=ax.transAxes,
               fontsize=28, fontweight="bold", color=color, ha="center", va="center")
        # Label
        ax.text(x + 0.08, 0.2, name, transform=ax.transAxes,
               fontsize=13, color="white", ha="center", va="center", alpha=0.8)
    
    ax.set_title("YOLO SKU-110K - Validation Metrics", fontsize=16, color="white", fontweight="bold", pad=20)
    save_path = os.path.join(save_dir, "yolo_metrics_card.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Metrics card saved: {save_path}")

plot_metrics_card(
    val_results.box.map50,
    val_results.box.map,
    val_results.box.mp,
    val_results.box.mr,
    FIG_DIR
)


# %% [markdown]
# ---
# # PART 2: DINOv2 SKU Embedding Index (FAISS)
# ---
# Uses the trained YOLO to detect products, then DINOv2 to create
# visual embeddings for SKU matching.

# %%
print("\n" + "="*60)
print("  PART 2: BUILDING DINOv2 SKU INDEX")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINOv2 ViT-Small (384-dim embeddings, ~85MB)
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2 = dinov2.to(device)
dinov2.eval()

# DINOv2 preprocessing
dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"DINOv2 loaded on {device}")

# %%
# === Find SKU-110K images (already downloaded by YOLO training) ===

def find_sku110k_images():
    """Find SKU-110K images that were auto-downloaded during YOLO training."""
    possible_paths = [
        # Ultralytics default download locations
        os.path.expanduser("~/datasets/SKU-110K/images"),
        "/kaggle/working/datasets/SKU-110K/images",
        "datasets/SKU-110K/images",
        # Kaggle working directory variants
        "/kaggle/working/SKU-110K/images",
    ]
    
    for base in possible_paths:
        for sub in ["train", "val", "test", ""]:
            path = os.path.join(base, sub) if sub else base
            if os.path.exists(path):
                imgs = glob.glob(os.path.join(path, "*.jpg"))
                if imgs:
                    print(f"  Found {len(imgs)} images in {path}")
                    return path, imgs
    
    # Fallback: search recursively from common roots
    for root in ["/kaggle/working", os.path.expanduser("~"), "."]:
        for dirpath, dirnames, filenames in os.walk(root):
            if "SKU-110K" in dirpath and "images" in dirpath:
                imgs = glob.glob(os.path.join(dirpath, "*.jpg"))
                if imgs:
                    print(f"  Found {len(imgs)} images in {dirpath}")
                    return dirpath, imgs
    
    return None, []


def find_supermarket_images():
    """Find Supermarket Shelves images if uploaded."""
    possible_roots = [
        SHELF_IMAGES_DIR,
        os.path.join(SHELF_IMAGES_DIR, "Supermarket shelves"),
        os.path.join(SHELF_IMAGES_DIR, "Supermarket shelves", "Supermarket shelves"),
    ]
    for root in possible_roots:
        img_dir = os.path.join(root, "images")
        if os.path.exists(img_dir):
            imgs = glob.glob(os.path.join(img_dir, "*.jpg"))
            if imgs:
                return img_dir, imgs
    return None, []

# %%
# === Extract product crops using trained YOLO ===

def extract_product_crops(yolo_model, image_files, max_crops_per_image=15, max_images=200):
    """Use trained YOLO to detect products, crop them for DINOv2 embedding."""
    crops = []
    metadata = []
    
    # Sample images evenly
    if len(image_files) > max_images:
        indices = np.linspace(0, len(image_files) - 1, max_images, dtype=int)
        image_files = [image_files[i] for i in indices]
    
    print(f"  Processing {len(image_files)} images for crop extraction...")
    
    for img_idx, img_path in enumerate(image_files):
        if img_idx % 50 == 0:
            print(f"    [{img_idx}/{len(image_files)}] Processing...")
        
        try:
            results = yolo_model(img_path, conf=0.3, verbose=False)
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            continue
        
        for det_idx, box in enumerate(results[0].boxes):
            if det_idx >= max_crops_per_image:
                break
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu())
            
            # Pad crop slightly
            pad = 3
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad)
            y2 = min(img.height, y2 + pad)
            
            # Skip tiny crops
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            
            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)
            metadata.append({
                "source_image": os.path.basename(img_path),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "crop_id": f"crop_{len(crops):05d}",
            })
    
    print(f"  Extracted {len(crops)} product crops")
    return crops, metadata

# Find images
print("Looking for SKU-110K images...")
sku_dir, sku_images = find_sku110k_images()
print("Looking for Supermarket Shelves images...")
sm_dir, sm_images = find_supermarket_images()

# Combine all available images
all_images = sku_images + sm_images
print(f"\nTotal images available: {len(all_images)} (SKU-110K: {len(sku_images)}, Shelves: {len(sm_images)})")

# Extract crops
det_model = YOLO(best_dst)
crops, crop_metadata = extract_product_crops(det_model, all_images, max_crops_per_image=15, max_images=200)

# %%
# === Build FAISS index from DINOv2 embeddings ===

import faiss

@torch.no_grad()
def extract_embeddings(model, crops, transform, device, batch_size=32):
    """Extract DINOv2 embeddings for all crops."""
    all_embeddings = []
    
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        batch_tensors = torch.stack([transform(c) for c in batch_crops]).to(device)
        embeddings = model(batch_tensors)
        all_embeddings.append(embeddings.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i // batch_size + 1}/{(len(crops) + batch_size - 1) // batch_size}")
    
    return np.vstack(all_embeddings)

print("\nExtracting DINOv2 embeddings...")
embeddings = extract_embeddings(dinov2, crops, dinov2_transform, device)
print(f"Embeddings shape: {embeddings.shape}")  # (N, 384)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index (Inner Product = cosine similarity after normalization)
embed_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embed_dim)
index.add(embeddings)

# Save
faiss_path = os.path.join(OUTPUT_DIR, "sku_faiss_index.bin")
faiss.write_index(index, faiss_path)

meta_path = os.path.join(OUTPUT_DIR, "sku_metadata.json")
with open(meta_path, "w") as f:
    json.dump(crop_metadata, f, indent=2)

# Save sample crop images for reference
crop_samples_dir = os.path.join(OUTPUT_DIR, "sample_crops")
os.makedirs(crop_samples_dir, exist_ok=True)
for i in range(min(50, len(crops))):
    crops[i].save(os.path.join(crop_samples_dir, f"crop_{i:05d}.jpg"))

print(f"\n[OK] FAISS index saved: {faiss_path}")
print(f"     {index.ntotal} vectors, {embed_dim}-dim")
print(f"[OK] Metadata saved: {meta_path}")
print(f"[OK] Sample crops saved: {crop_samples_dir}")

# %% [markdown]
# ## VISUALIZATION 2: DINOv2 Embeddings & Sample Crops

# %%
# === VIZ 2A: Sample Crop Grid ===

def plot_crop_grid(crops_list, save_dir, n_show=40):
    """Show a grid of detected product crops."""
    n_show = min(n_show, len(crops_list))
    cols = 10
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2.2 * rows))
    fig.suptitle(f"ShelfMind AI - Detected Product Crops ({len(crops_list)} total)",
                fontsize=16, color="white", fontweight="bold")
    
    for i in range(rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        if i < n_show:
            ax.imshow(crops_list[i])
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(save_dir, "dinov2_crop_grid.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Crop grid saved: {save_path}")

plot_crop_grid(crops, FIG_DIR)

# %%
# === VIZ 2B: t-SNE of DINOv2 Embeddings ===

def plot_embedding_tsne(emb, metadata, save_dir, n_max=1000):
    """Visualize DINOv2 embeddings with t-SNE."""
    from sklearn.manifold import TSNE
    
    n = min(n_max, len(emb))
    subset_emb = emb[:n]
    
    print(f"  Running t-SNE on {n} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1), n_iter=1000)
    coords = tsne.fit_transform(subset_emb)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color by confidence
    confs = [m["confidence"] for m in metadata[:n]]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=confs, cmap="viridis",
                        s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Detection Confidence", shrink=0.8)
    
    ax.set_title(f"DINOv2 Product Embeddings (t-SNE, {n} crops)",
                fontsize=16, color="white", fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=12)
    ax.set_ylabel("t-SNE dim 2", fontsize=12)
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "dinov2_tsne.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] t-SNE saved: {save_path}")

if len(embeddings) > 10:
    plot_embedding_tsne(embeddings, crop_metadata, FIG_DIR)

# %%
# === VIZ 2C: FAISS Nearest Neighbor Demo ===

def plot_faiss_nn_demo(faiss_index, crops_list, emb, save_dir, n_queries=4, k=5):
    """Show nearest neighbor retrieval: query crop -> top-k similar crops."""
    if len(crops_list) < k + n_queries:
        print("  [SKIP] Not enough crops for NN demo")
        return
    
    # Pick random query crops
    np.random.seed(123)
    query_ids = np.random.choice(len(crops_list), n_queries, replace=False)
    
    fig, axes = plt.subplots(n_queries, k + 1, figsize=(3 * (k + 1), 3.5 * n_queries))
    fig.suptitle("FAISS Nearest Neighbor Retrieval Demo",
                fontsize=16, color="white", fontweight="bold")
    
    for row, qid in enumerate(query_ids):
        # Query
        query_emb = emb[qid:qid+1].copy()
        distances, indices = faiss_index.search(query_emb, k + 1)  # +1 because first match is itself
        
        ax = axes[row, 0] if n_queries > 1 else axes[0]
        ax.imshow(crops_list[qid])
        ax.set_title("QUERY", fontsize=11, color=COLORS[0], fontweight="bold")
        ax.spines[:].set_color(COLORS[0])
        ax.spines[:].set_linewidth(3)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Neighbors (skip index 0 = self)
        for col_idx in range(k):
            nn_idx = indices[0, col_idx + 1] if col_idx + 1 < len(indices[0]) else indices[0, col_idx]
            sim = distances[0, col_idx + 1] if col_idx + 1 < len(distances[0]) else distances[0, col_idx]
            
            ax = axes[row, col_idx + 1] if n_queries > 1 else axes[col_idx + 1]
            ax.imshow(crops_list[nn_idx])
            ax.set_title(f"Sim: {sim:.2f}", fontsize=10, color="white")
            ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(save_dir, "faiss_nn_demo.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] NN demo saved: {save_path}")

plot_faiss_nn_demo(index, crops, embeddings, FIG_DIR)

# Free GPU memory
del dinov2
torch.cuda.empty_cache()


# %% [markdown]
# ---
# # PART 3: Demand Forecasting (LightGBM on M5 Walmart Sales)
# ---
# M5 dataset: 30,490 products x 1,913 days across 10 stores (CA/TX/WI)

# %%
print("\n" + "="*60)
print("  PART 3: TRAINING DEMAND FORECASTING MODEL")
print("="*60)

# Find M5 data
m5_paths = [M5_DATA_DIR, "/kaggle/input/competitions/m5-forecasting-accuracy", "/kaggle/input/m5-forecasting-accuracy", "data/m5-forecasting-accuracy"]
m5_dir = None
for p in m5_paths:
    if os.path.exists(p):
        m5_dir = p
        break

if m5_dir is None:
    raise FileNotFoundError("M5 dataset not found! Add 'm5-forecasting-accuracy' dataset on Kaggle.")

print(f"Loading M5 from: {m5_dir}")

# Load data
calendar = pd.read_csv(os.path.join(m5_dir, "calendar.csv"))
print(f"  Calendar: {calendar.shape}")

prices = pd.read_csv(os.path.join(m5_dir, "sell_prices.csv"))
print(f"  Prices: {prices.shape}")

print("  Loading sales data...")
sales = pd.read_csv(os.path.join(m5_dir, "sales_train_validation.csv"))
print(f"  Sales: {sales.shape}")

# %%
# === Feature Engineering ===

def prepare_forecast_data(sales_df, calendar_df, prices_df, n_items=500, last_n_days=365):
    """Convert wide M5 format to long format with features."""
    
    # Get day columns
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    
    # Select top-selling items for speed
    sales_df = sales_df.copy()
    sales_df["total_sales"] = sales_df[day_cols[-last_n_days:]].sum(axis=1)
    top_items = sales_df.nlargest(n_items, "total_sales")
    
    # Use last N days
    use_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    use_day_cols = day_cols[-last_n_days:]
    
    # Melt to long format
    print(f"  Converting {n_items} items x {last_n_days} days...")
    long_df = top_items[use_cols + use_day_cols].melt(
        id_vars=use_cols,
        value_vars=use_day_cols,
        var_name="d",
        value_name="sales"
    )
    
    # Merge calendar
    cal_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    long_df = long_df.merge(calendar_df[cal_cols], on="d", how="left")
    long_df["date"] = pd.to_datetime(long_df["date"])
    
    # Merge prices
    long_df = long_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    # === Feature Engineering ===
    long_df["day_of_week"] = long_df["date"].dt.dayofweek
    long_df["day_of_month"] = long_df["date"].dt.day
    long_df["week_of_year"] = long_df["date"].dt.isocalendar().week.astype(int)
    long_df["is_weekend"] = (long_df["day_of_week"] >= 5).astype(int)
    long_df["is_month_start"] = (long_df["day_of_month"] <= 3).astype(int)
    long_df["is_month_end"] = (long_df["day_of_month"] >= 28).astype(int)
    
    # Events
    long_df["has_event"] = long_df["event_name_1"].notna().astype(int)
    
    # SNAP
    long_df["snap"] = 0
    for state in ["CA", "TX", "WI"]:
        mask = long_df["state_id"] == state
        long_df.loc[mask, "snap"] = long_df.loc[mask, f"snap_{state}"]
    
    # Lag features
    print("  Computing lag features...")
    long_df = long_df.sort_values(["id", "date"])
    
    for lag in [7, 14, 28]:
        long_df[f"sales_lag_{lag}"] = long_df.groupby("id")["sales"].shift(lag)
    
    for window in [7, 14, 28]:
        long_df[f"sales_rolling_mean_{window}"] = (
            long_df.groupby("id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        long_df[f"sales_rolling_std_{window}"] = (
            long_df.groupby("id")["sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    
    # Encode categoricals
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        long_df[col + "_enc"] = long_df[col].astype("category").cat.codes
    
    long_df = long_df.dropna()
    print(f"  Final dataset: {long_df.shape}")
    return long_df


forecast_df = prepare_forecast_data(sales, calendar, prices, n_items=500, last_n_days=365)

# %%
# === Train LightGBM ===

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

FEATURES = [
    "sell_price",
    "day_of_week", "day_of_month", "week_of_year", "month",
    "is_weekend", "is_month_start", "is_month_end",
    "has_event", "snap",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28",
    "sales_rolling_std_7", "sales_rolling_std_14", "sales_rolling_std_28",
    "item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc", "state_id_enc",
]

TARGET = "sales"

# Time-based split (last 28 days = test)
forecast_df = forecast_df.sort_values("date")
split_date = forecast_df["date"].max() - pd.Timedelta(days=28)

train_df = forecast_df[forecast_df["date"] <= split_date]
test_df = forecast_df[forecast_df["date"] > split_date]

print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# Train
print("\nTraining LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

# Evaluate
y_pred = np.maximum(lgb_model.predict(X_test), 0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs(y_test - y_pred) / np.maximum(y_test, 1)) * 100

print(f"\n--- Forecast Results ---")
print(f"  MAE:  {mae:.3f}")
print(f"  RMSE: {rmse:.3f}")
print(f"  MAPE: {mape:.1f}%")

# Feature importance
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print(f"\nTop 10 Features:")
print(importance.head(10).to_string(index=False))

# Save
model_path = os.path.join(OUTPUT_DIR, "lgbm_forecast_model.pkl")
joblib.dump({
    "model": lgb_model,
    "features": FEATURES,
    "target": TARGET,
    "metrics": {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)},
    "feature_importance": importance.to_dict("records"),
}, model_path)

print(f"\n[OK] LightGBM saved: {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB)")

# %% [markdown]
# ## VISUALIZATION 3: Demand Forecasting Results

# %%
# === VIZ 3A: Feature Importance Bar Chart ===

def plot_feature_importance(importance_df, save_dir):
    """Horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    top = importance_df.head(15).sort_values("importance")
    bars = ax.barh(top["feature"], top["importance"], color=COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.5)
    
    # Highlight top 3
    for i, bar in enumerate(bars):
        if i >= len(bars) - 3:
            bar.set_color(COLORS[3])
            bar.set_alpha(1.0)
    
    ax.set_title("LightGBM Feature Importance (Top 15)", fontsize=16, color="white", fontweight="bold")
    ax.set_xlabel("Importance (Split Count)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.15)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "lgbm_feature_importance.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Feature importance saved: {save_path}")

plot_feature_importance(importance, FIG_DIR)

# %%
# === VIZ 3B: Actual vs Predicted Scatter + Residuals ===

def plot_actual_vs_predicted(y_true, y_predicted, mae_val, rmse_val, save_dir):
    """Scatter plot of actual vs predicted sales with residual distribution."""
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    
    # Panel 1: Scatter
    ax1 = fig.add_subplot(gs[0])
    sample_size = min(5000, len(y_true))
    idx = np.random.choice(len(y_true), sample_size, replace=False)
    y_t = np.array(y_true)[idx]
    y_p = np.array(y_predicted)[idx]
    
    ax1.scatter(y_t, y_p, alpha=0.3, s=10, color=COLORS[0], edgecolors="none")
    max_val = max(y_t.max(), y_p.max())
    ax1.plot([0, max_val], [0, max_val], "--", color=COLORS[1], linewidth=2, label="Perfect prediction")
    ax1.set_title("Actual vs Predicted Sales", fontsize=14, color="white", fontweight="bold")
    ax1.set_xlabel("Actual Sales", fontsize=12)
    ax1.set_ylabel("Predicted Sales", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15)
    
    # Panel 2: Residual Distribution
    ax2 = fig.add_subplot(gs[1])
    residuals = np.array(y_true) - np.array(y_predicted)
    ax2.hist(residuals, bins=50, color=COLORS[2], alpha=0.75, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color=COLORS[1], linewidth=2, linestyle="--")
    ax2.set_title("Prediction Residuals", fontsize=14, color="white", fontweight="bold")
    ax2.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.grid(True, alpha=0.15)
    
    # Panel 3: Metrics Card
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    metrics_text = [
        ("MAE", f"{mae_val:.3f}", COLORS[0]),
        ("RMSE", f"{rmse_val:.3f}", COLORS[1]),
        ("Median AE", f"{np.median(np.abs(residuals)):.3f}", COLORS[2]),
        ("% < 1 unit", f"{(np.abs(residuals) < 1).mean() * 100:.1f}%", COLORS[3]),
    ]
    for i, (name, value, color) in enumerate(metrics_text):
        y_pos = 0.8 - i * 0.2
        ax3.text(0.15, y_pos, name + ":", transform=ax3.transAxes,
                fontsize=16, color="white", alpha=0.7, va="center")
        ax3.text(0.75, y_pos, value, transform=ax3.transAxes,
                fontsize=22, color=color, fontweight="bold", va="center", ha="center")
    ax3.set_title("Error Metrics", fontsize=14, color="white", fontweight="bold", pad=15)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "lgbm_predictions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Predictions plot saved: {save_path}")

plot_actual_vs_predicted(y_test.values, y_pred, mae, rmse, FIG_DIR)

# %%
# === VIZ 3C: Sales Forecast Timeline (sample items) ===

def plot_forecast_timeline(test_data, y_predicted, save_dir, n_items=4):
    """Plot actual vs predicted sales over time for sample items."""
    test_data = test_data.copy()
    test_data["predicted"] = y_predicted
    
    # Pick top items by total sales
    item_sales = test_data.groupby("id")["sales"].sum().nlargest(n_items)
    sample_items = item_sales.index.tolist()
    
    fig, axes = plt.subplots(n_items, 1, figsize=(16, 3.5 * n_items), sharex=True)
    fig.suptitle("Demand Forecast vs Actual (Top Items)", fontsize=16, color="white", fontweight="bold")
    
    for idx, item_id in enumerate(sample_items):
        ax = axes[idx] if n_items > 1 else axes
        item_data = test_data[test_data["id"] == item_id].sort_values("date")
        
        ax.plot(item_data["date"], item_data["sales"], color=COLORS[0],
               linewidth=2, label="Actual", alpha=0.9)
        ax.plot(item_data["date"], item_data["predicted"], color=COLORS[1],
               linewidth=2, linestyle="--", label="Predicted", alpha=0.9)
        ax.fill_between(item_data["date"], item_data["sales"], item_data["predicted"],
                       alpha=0.1, color=COLORS[1])
        
        item_mae = np.abs(item_data["sales"] - item_data["predicted"]).mean()
        short_id = item_id.split("_validation")[0] if "_validation" in item_id else item_id[:30]
        ax.set_title(f"{short_id} (MAE: {item_mae:.2f})", fontsize=11, color="white")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, "lgbm_forecast_timeline.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()
    print(f"  [OK] Forecast timeline saved: {save_path}")

plot_forecast_timeline(test_df, y_pred, FIG_DIR)


# %% [markdown]
# ---
# # PART 4: Package All Models
# ---

# %%
import zipfile

print("\n" + "="*60)
print("   SHELFMIND AI - ALL TRAINING COMPLETE!")
print("="*60)

model_files = [
    ("yolo_shelf_best.pt", "Product Detection (YOLO26s trained on SKU-110K — NMS-free)"),
    ("sku_faiss_index.bin", "SKU Matching (DINOv2 + FAISS)"),
    ("sku_metadata.json", "SKU Crop Metadata"),
    ("lgbm_forecast_model.pkl", "Demand Forecasting (LightGBM on M5)"),
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

print(f"\n  Total: {total_size:.1f} MB")

# Create zip for easy download
zip_path = os.path.join(OUTPUT_DIR, "shelfmind_models.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename, _ in model_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            zf.write(filepath, filename)
    
    # Also include sample crops
    crop_dir = os.path.join(OUTPUT_DIR, "sample_crops")
    if os.path.exists(crop_dir):
        for crop_file in glob.glob(os.path.join(crop_dir, "*.jpg"))[:50]:
            zf.write(crop_file, f"sample_crops/{os.path.basename(crop_file)}")
    
    # Include all visualization charts
    if os.path.exists(FIG_DIR):
        for fig_file in glob.glob(os.path.join(FIG_DIR, "*.png")):
            zf.write(fig_file, f"visualizations/{os.path.basename(fig_file)}")

zip_size = os.path.getsize(zip_path) / 1e6
print(f"\n  [OK] All-in-one zip: {zip_path} ({zip_size:.1f} MB)")

print(f"\n{'='*60}")
print("  NEXT STEPS:")
print("  1. Click 'Save Version' to run the notebook")
print("  2. Go to Output tab -> download shelfmind_models.zip")
print("  3. Extract to: Smart Retail Shelf Intelligence/models/")
print("  4. We'll build the Streamlit dashboard on your PC!")
print(f"{'='*60}")
