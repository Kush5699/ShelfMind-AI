"""
=============================================================================
ShelfMind AI — RF-DETR Fine-Tuning on Lightning.ai
=============================================================================
Run on Lightning.ai with GPU (L4 24GB / A10G 24GB / A100 40GB)

SETUP:
1. Create a new Studio on Lightning.ai → Select GPU (L4 recommended)
2. Upload this single script
3. Run: python rfdetr_lightning.py

OUTPUT:
  - rfdetr_shelf_best.pt       (~130 MB) → Fine-tuned RF-DETR detector
  - extracted_crops.zip         (~2 GB)  → Crops for DINOv2 fine-tuning
  - rfdetr_metrics.json                  → Validation metrics
=============================================================================
"""

# =============================================================================
# STEP 0: Install all dependencies (handles Lightning.ai environment cleanly)
# =============================================================================
import subprocess
import sys
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Suppress version check warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce OOM fragmentation

def run_pip(*args):
    """Run pip with proper error handling."""
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(args)
    subprocess.check_call(cmd)

# Install rfdetr FIRST — it pulls compatible albumentations
run_pip("-U", "rfdetr")
# Ensure albumentations is in the supported range (1.4.24+ or 2.x)
run_pip("-U", "albumentations>=1.4.24")
# Additional deps
run_pip("faster-coco-eval", "pycocotools", "supervision")

print("[OK] All dependencies installed")

# =============================================================================
# STEP 0b: Imports + Logging + GPU Check
# =============================================================================
import json
import glob
import shutil
import gc
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # SKU-110K has some truncated images

import torch

# --- Output directory ---
OUTPUT_DIR = "shelfmind_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging: Tee all output to log.txt ---
class TeeLogger:
    """Writes output to both console and log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self.log_file.write(f"=== ShelfMind AI — RF-DETR Training Log ===\n")
        self.log_file.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

LOG_PATH = os.path.join(OUTPUT_DIR, "log.txt")
sys.stdout = TeeLogger(LOG_PATH)
sys.stderr = TeeLogger(os.path.join(OUTPUT_DIR, "error_log.txt"))
print(f"[LOG] All output saved to: {LOG_PATH}")

# --- GPU check ---
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "ERROR: No GPU found! Select a GPU machine on Lightning.ai."

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")
print(f"Output: {OUTPUT_DIR}\n")


# =============================================================================
# STEP 1: Download SKU-110K + Convert to COCO Format
# =============================================================================
print("=" * 60)
print("  STEP 1: DOWNLOAD SKU-110K DATASET")
print("=" * 60)

SKU_DIR = "datasets/SKU-110K"
TAR_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"

os.makedirs("datasets", exist_ok=True)

if not os.path.exists(SKU_DIR):
    print("  Downloading + extracting SKU-110K (~8 min)...")
    subprocess.run(
        f"curl -L {TAR_URL} | tar -xz -C datasets/",
        shell=True, check=True
    )
    # Handle different extraction names
    for candidate_name in ["datasets/SKU110K_fixed", "datasets/SKU110K"]:
        if os.path.exists(candidate_name) and not os.path.exists(SKU_DIR):
            shutil.move(candidate_name, SKU_DIR)  # shutil.move works across filesystems
            break
    print(f"  [OK] SKU-110K extracted!")
    os.system("df -h . | tail -1")
else:
    print(f"  SKU-110K already exists at {SKU_DIR}")

# Verify dataset structure
ann_dir = os.path.join(SKU_DIR, "annotations")
images_dir = os.path.join(SKU_DIR, "images")
assert os.path.exists(ann_dir), f"ERROR: annotations dir not found: {ann_dir}"
assert os.path.exists(images_dir), f"ERROR: images dir not found: {images_dir}"
n_images = len(glob.glob(os.path.join(images_dir, "*.jpg")))
print(f"  Verified: {n_images} images, annotations dir present\n")


# --- Convert to COCO format ---
print("  Converting annotations to COCO JSON format...")

def convert_sku110k_to_coco(sku_dir):
    """Convert SKU-110K CSV annotations to COCO JSON for RF-DETR."""
    ann_dir_local = os.path.join(sku_dir, "annotations")
    coco_base = os.path.join(sku_dir, "coco_format")
    os.makedirs(coco_base, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        csv_file = os.path.join(ann_dir_local, f"annotations_{split_name}.csv")
        if not os.path.exists(csv_file):
            print(f"    [SKIP] {csv_file} not found")
            continue

        # RF-DETR expects "valid" not "val"
        folder_name = "valid" if split_name == "val" else split_name

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
                x1, y1 = float(row["x1"]), float(row["y1"])
                x2, y2 = float(row["x2"]), float(row["y2"])
                w, h = abs(x2 - x1), abs(y2 - y1)
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

        # Link images into the split folder
        src_images = os.path.join(sku_dir, "images")
        for img_info in coco["images"]:
            src = os.path.join(src_images, img_info["file_name"])
            dst = os.path.join(split_dir, img_info["file_name"])
            if os.path.exists(src) and not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)

        json_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(json_path, "w") as f:
            json.dump(coco, f)

        print(f"    {folder_name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    print(f"  [OK] COCO format ready: {coco_base}")
    return coco_base


coco_dataset_dir = convert_sku110k_to_coco(SKU_DIR)


# =============================================================================
# STEP 2: Train RF-DETR-Base on SKU-110K
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2: RF-DETR-Base TRAINING ON SKU-110K")
print("  DINOv2 backbone | Transformer attention | ICLR 2026")
print("=" * 60)

from rfdetr import RFDETRBase

# Auto-select batch size based on VRAM
# NOTE: SKU-110K is extremely dense (~147 bboxes/image).
# The GIoU loss creates 300×147 matching matrices that scale with
# batch_size × resolution. batch=4 OOMs even on L4 at 840px multi-scale.
if vram_gb >= 40:      # A100 (40/80GB)
    BATCH_SIZE, GRAD_ACCUM = 4, 4
elif vram_gb >= 22:    # L4 / A10G (24GB)
    BATCH_SIZE, GRAD_ACCUM = 2, 8
else:
    raise RuntimeError(
        f"GPU has only {vram_gb:.1f}GB VRAM. RF-DETR on SKU-110K needs 22GB+.\n"
        f"Please select an L4 or A100 GPU on Lightning.ai."
    )

print(f"\n  GPU:          {gpu_name}")
print(f"  VRAM:         {vram_gb:.1f} GB")
print(f"  Backbone:     DINOv2 ViT (pretrained)")
print(f"  batch_size:   {BATCH_SIZE}")
print(f"  grad_accum:   {GRAD_ACCUM}")
print(f"  eff. batch:   {BATCH_SIZE * GRAD_ACCUM}")
print(f"  epochs:       30")
print(f"  lr:           1e-4\n")

rfdetr_model = RFDETRBase()

rfdetr_model.train(
    dataset_dir=coco_dataset_dir,
    epochs=30,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM,
    lr=1e-4,
    output_dir=os.path.join(OUTPUT_DIR, "rfdetr_sku110k"),
)

print("\n[OK] RF-DETR-Base training complete!")

# Free training model VRAM
del rfdetr_model
torch.cuda.empty_cache()
gc.collect()


# =============================================================================
# STEP 3: Save Best Model + Validate
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3: SAVE & VALIDATE RF-DETR MODEL")
print("=" * 60)

rfdetr_output_dir = os.path.join(OUTPUT_DIR, "rfdetr_sku110k")
rfdetr_best_dst = os.path.join(OUTPUT_DIR, "rfdetr_shelf_best.pt")

# Find best checkpoint (try multiple naming conventions)
rfdetr_ckpt = None
search_patterns = [
    "best_checkpoint.pth", "checkpoint_best.pth",
    "best.pth", "model_best.pth", "last.pth",
]
for name in search_patterns:
    candidate = os.path.join(rfdetr_output_dir, name)
    if os.path.exists(candidate):
        rfdetr_ckpt = candidate
        break

# Fallback: find any .pth file
if rfdetr_ckpt is None:
    pth_files = sorted(glob.glob(os.path.join(rfdetr_output_dir, "**", "*.pth"), recursive=True))
    if pth_files:
        rfdetr_ckpt = pth_files[-1]  # Take the last (usually best or latest)

if rfdetr_ckpt:
    shutil.copy2(rfdetr_ckpt, rfdetr_best_dst)
    size_mb = os.path.getsize(rfdetr_best_dst) / 1e6
    print(f"\n[OK] Model saved: {rfdetr_best_dst} ({size_mb:.1f} MB)")
    print(f"     Source: {rfdetr_ckpt}")
else:
    print("[ERROR] No checkpoint found!")
    print("  Listing output directory:")
    for f in glob.glob(os.path.join(rfdetr_output_dir, "**", "*"), recursive=True):
        print(f"  {f}")

# --- Validate on val set ---
print("\nValidating RF-DETR-Base on SKU-110K val set...")

rfdetr_metrics = {"model": "RF-DETR-Base", "mAP50": 0, "mAP50_95": 0,
                  "precision": 0, "recall": 0, "size_MB": 0}

try:
    val_coco_dir = os.path.join(coco_dataset_dir, "valid")
    val_json = os.path.join(val_coco_dir, "_annotations.coco.json")

    if os.path.exists(val_json):
        with open(val_json) as f:
            val_coco = json.load(f)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt_coco = COCO(val_json)

        # Load fine-tuned model for inference
        rfdetr_eval = RFDETRBase()
        if rfdetr_ckpt:
            print(f"  Loading fine-tuned weights from: {rfdetr_ckpt}")
            ckpt = torch.load(rfdetr_ckpt, map_location="cuda", weights_only=False)
            state = ckpt.get("ema_model", ckpt.get("model", ckpt.get("state_dict", ckpt)))
            # Handle Lightning checkpoint format
            if any(k.startswith("model.") for k in state.keys()):
                state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
            rfdetr_eval.model.load_state_dict(state, strict=False)
            print(f"  [OK] Fine-tuned weights loaded")

        predictions = []
        n_images = len(val_coco["images"])
        print(f"  Running predictions on {n_images} val images...")

        for idx, img_info in enumerate(val_coco["images"]):
            img_path = os.path.join(val_coco_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            if idx % 100 == 0:
                print(f"    [{idx}/{n_images}]...")
            try:
                # predict() returns supervision.Detections object
                detections = rfdetr_eval.predict(img_path, threshold=0.3)
                if detections is not None and len(detections) > 0:
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        predictions.append({
                            "image_id": img_info["id"],
                            "category_id": 0,
                            "bbox": [float(x1), float(y1), w, h],
                            "score": float(detections.confidence[i]),
                        })
            except Exception:
                continue

        print(f"  Total predictions: {len(predictions)}")

        if predictions:
            pred_coco = gt_coco.loadRes(predictions)
            coco_eval = COCOeval(gt_coco, pred_coco, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            rfdetr_metrics = {
                "model": "RF-DETR-Base",
                "mAP50": float(coco_eval.stats[1]),
                "mAP50_95": float(coco_eval.stats[0]),
                "precision": float(coco_eval.stats[8]) if len(coco_eval.stats) > 8 else 0.0,
                "recall": float(coco_eval.stats[6]) if len(coco_eval.stats) > 6 else 0.0,
                "size_MB": os.path.getsize(rfdetr_best_dst) / 1e6 if os.path.exists(rfdetr_best_dst) else 0,
            }

        del rfdetr_eval
        torch.cuda.empty_cache()
        gc.collect()

except Exception as e:
    print(f"  [WARNING] Evaluation error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n--- RF-DETR-Base Validation Results ---")
for k, v in rfdetr_metrics.items():
    print(f"  {k}: {v}")

# Save metrics
metrics_path = os.path.join(OUTPUT_DIR, "rfdetr_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(rfdetr_metrics, f, indent=2)
print(f"[OK] Metrics saved: {metrics_path}")


# =============================================================================
# STEP 4: Extract Product Crops for DINOv2 Fine-Tuning
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4: EXTRACT PRODUCT CROPS FOR DINOv2 FINE-TUNING")
print("=" * 60)

# Load fine-tuned model for crop extraction
crop_model = RFDETRBase()
if rfdetr_ckpt:
    print(f"  Loading fine-tuned weights from: {rfdetr_ckpt}")
    ckpt = torch.load(rfdetr_ckpt, map_location="cuda", weights_only=False)
    state = ckpt.get("ema_model", ckpt.get("model", ckpt.get("state_dict", ckpt)))
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
    crop_model.model.load_state_dict(state, strict=False)
    print(f"  [OK] Fine-tuned weights loaded")
else:
    print("  [WARNING] No fine-tuned model available, using pretrained")

# Find all images
all_images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
print(f"  Found {len(all_images)} total images")

crops_dir = os.path.join(OUTPUT_DIR, "crops_for_dinov2")
os.makedirs(crops_dir, exist_ok=True)

metadata = []
crop_count = 0
MAX_CROPS_PER_IMAGE = 20

print(f"  Extracting crops (max {MAX_CROPS_PER_IMAGE}/image)...")

for img_idx, img_path in enumerate(all_images):
    if img_idx % 500 == 0:
        print(f"    [{img_idx}/{len(all_images)}] {crop_count} crops so far...")

    try:
        detections = crop_model.predict(img_path, threshold=0.35)
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue

    source_name = os.path.basename(img_path)

    if detections is None or len(detections) == 0:
        continue

    for det_idx in range(min(len(detections), MAX_CROPS_PER_IMAGE)):
        x1, y1, x2, y2 = detections.xyxy[det_idx].astype(int)
        conf = float(detections.confidence[det_idx])

        if conf < 0.35:
            continue

        # Pad slightly for better crops
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
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        })
        crop_count += 1

del crop_model
torch.cuda.empty_cache()
gc.collect()

# Save metadata
meta_path = os.path.join(OUTPUT_DIR, "crop_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f)

print(f"\n  [OK] Extracted {crop_count} product crops")
print(f"  [OK] Crops dir: {crops_dir}")
print(f"  [OK] Metadata: {meta_path}")


# =============================================================================
# STEP 5: Package for Download
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5: PACKAGING OUTPUTS")
print("=" * 60)

import zipfile

# RF-DETR model zip
rfdetr_zip = os.path.join(OUTPUT_DIR, "rfdetr_shelf.zip")
with zipfile.ZipFile(rfdetr_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(rfdetr_best_dst):
        zf.write(rfdetr_best_dst, "rfdetr_shelf_best.pt")
    if os.path.exists(metrics_path):
        zf.write(metrics_path, "rfdetr_metrics.json")
    if os.path.exists(LOG_PATH):
        zf.write(LOG_PATH, "training_log.txt")
rz = os.path.getsize(rfdetr_zip) / 1e6
print(f"  [OK] rfdetr_shelf.zip        ({rz:.1f} MB)")

# Crops zip
crops_zip = os.path.join(OUTPUT_DIR, "extracted_crops.zip")
with zipfile.ZipFile(crops_zip, "w", zipfile.ZIP_STORED) as zf:
    crop_files = sorted(glob.glob(os.path.join(crops_dir, "*.jpg")))
    for i, crop_file in enumerate(crop_files):
        zf.write(crop_file, f"crops/{os.path.basename(crop_file)}")
        if i % 10000 == 0 and i > 0:
            print(f"    Zipped {i}/{len(crop_files)} crops...")
    if os.path.exists(meta_path):
        zf.write(meta_path, "crop_metadata.json")
cz = os.path.getsize(crops_zip) / 1e6
print(f"  [OK] extracted_crops.zip     ({cz:.1f} MB, {len(crop_files)} crops)")


# =============================================================================
# DONE!
# =============================================================================
print("\n" + "=" * 60)
print("   SHELFMIND AI — RF-DETR TRAINING COMPLETE!")
print("=" * 60)

print(f"\n--- RF-DETR-Base Results ---")
for k, v in rfdetr_metrics.items():
    print(f"  {k}: {v}")

print(f"\n--- Output Files ---")
print(f"  rfdetr_shelf.zip        ({rz:.1f} MB)  — Fine-tuned RF-DETR weights")
print(f"  extracted_crops.zip     ({cz:.1f} MB)  — {crop_count} crops for DINOv2")
print(f"  log.txt                           — Full training log")

print(f"\n--- Compare with YOLO26s (Kaggle) ---")
print(f"  YOLO26s mAP50:     0.895  (from previous Kaggle run)")
print(f"  RF-DETR mAP50:     {rfdetr_metrics['mAP50']:.4f}  (this run)")

end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"\nFinished: {end_time}")

print(f"\n{'=' * 60}")
print("  NEXT STEPS:")
print("  1. Download rfdetr_shelf.zip + extracted_crops.zip")
print("  2. Run dinov2_finetune.py (separate session)")
print("  3. Integrate models into web app")
print(f"{'=' * 60}")
