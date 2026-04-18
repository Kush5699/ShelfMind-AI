"""
=============================================================================
ShelfMind AI — YOLO26s Crop Extraction + DINOv2 Contrastive Fine-Tuning
=============================================================================
Kaggle Notebook Script (GPU P100/T4/L4 required)

SETUP:
  1. Upload yolo_shelf_best.pt as a Kaggle dataset (e.g., "shelfmind-yolo26")
  2. Add it as input to your notebook
  3. Enable GPU accelerator (T4 recommended)
  4. Paste this entire script into a notebook cell and run

  NOTE: SKU-110K is auto-downloaded (13.6 GB streaming — no double disk usage)
        Only the YOLO model needs to be uploaded as Kaggle input!

OUTPUT (in /kaggle/working/):
  - crops_for_dinov2/           → Extracted product crops
  - dinov2_shelf_finetuned.pth  → Fine-tuned DINOv2 backbone
  - dinov2_projector.pth        → Projection head
  - dinov2_shelf.zip            → Packaged for download
  - crop_metadata.json          → Crop bounding box metadata
=============================================================================
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================
import subprocess, sys, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def pip_install(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + list(pkgs),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# Install ultralytics for YOLO26s
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3", "timm", "tqdm"],
)
print("[OK] Dependencies installed")

# =============================================================================
# CELL 2: Imports & Config
# =============================================================================
import json
import glob
import gc
import math
import shutil
import zipfile
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile, ImageFilter
import numpy as np
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Paths ---
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)

# --- Auto-detect YOLO model path ---
YOLO_MODEL_CANDIDATES = [
    "/kaggle/input/models/kushpatel7391/shelfmind-yolo26/pytorch/default/1/yolo_shelf_best.pt",
    "/kaggle/input/shelfmind-yolo26/yolo_shelf_best.pt",
    "/kaggle/input/shelfmind-models/yolo_shelf_best.pt",
    "/kaggle/input/shelfmind-yolo/yolo_shelf_best.pt",
    "/kaggle/input/yolo-shelf-best/yolo_shelf_best.pt",
    "/kaggle/input/yolo26-shelfmind/yolo_shelf_best.pt",
]
# Also search all input directories for any .pt file matching the name
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "yolo_shelf_best.pt":
            YOLO_MODEL_CANDIDATES.insert(0, os.path.join(root, f))

YOLO_MODEL = None
for c in YOLO_MODEL_CANDIDATES:
    if os.path.exists(c):
        YOLO_MODEL = c
        break

assert YOLO_MODEL is not None, (
    "yolo_shelf_best.pt not found! Upload it as a Kaggle dataset input.\n"
    f"Searched: {YOLO_MODEL_CANDIDATES[:3]}"
)
print(f"[OK] YOLO model: {YOLO_MODEL} ({os.path.getsize(YOLO_MODEL)/1e6:.1f} MB)")

# --- Auto-detect SKU-110K or download ---
SKU_DIR = "/kaggle/working/datasets/SKU-110K"
TAR_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
os.makedirs("/kaggle/working/datasets", exist_ok=True)

if not os.path.exists(SKU_DIR):
    print(f"  Downloading + extracting SKU-110K (streaming, ~8 min)...")
    print(f"  (tar.gz streams directly to extraction — no double disk usage)")
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

# Find images directory
if os.path.exists(os.path.join(SKU_DIR, "images")):
    IMAGES_DIR = os.path.join(SKU_DIR, "images")
else:
    IMAGES_DIR = SKU_DIR

all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"[OK] SKU-110K: {SKU_DIR}")
print(f"[OK] Found {len(all_images)} images in {IMAGES_DIR}")

# --- GPU ---
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU required! Enable GPU in Kaggle settings."
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")


# =============================================================================
# CELL 3: YOLO26s Product Crop Extraction
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 1: YOLO26s PRODUCT CROP EXTRACTION")
print("=" * 60)

from ultralytics import YOLO

# Load fine-tuned YOLO26s model
print(f"  Loading YOLO26s from: {YOLO_MODEL}")
yolo_model = YOLO(YOLO_MODEL)
print(f"  [OK] YOLO26s loaded")

# Output directories
CROPS_DIR = os.path.join(WORKING_DIR, "crops_for_dinov2")
os.makedirs(CROPS_DIR, exist_ok=True)

# --- Extract crops ---
MAX_CROPS_PER_IMAGE = 10   # Limit to top-10 per image (saves disk space)
MIN_CROP_SIZE = 15         # Minimum crop width/height in pixels
CONFIDENCE_THRESHOLD = 0.35
PADDING = 3                # Pixel padding around crops

metadata = []
crop_count = 0

print(f"  Processing {len(all_images)} images...")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  Max crops/image: {MAX_CROPS_PER_IMAGE}")

for img_idx, img_path in enumerate(all_images):
    if img_idx % 500 == 0:
        print(f"    [{img_idx}/{len(all_images)}] — {crop_count} crops so far...")

    try:
        # YOLO inference
        results = yolo_model.predict(
            img_path,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            imgsz=640,
        )

        if not results or len(results[0].boxes) == 0:
            continue

        # Open image for cropping
        img = Image.open(img_path).convert("RGB")
        boxes = results[0].boxes

        # Sort by confidence (highest first)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        sort_idx = np.argsort(-confs)

        source = os.path.basename(img_path)

        for det_i in range(min(len(sort_idx), MAX_CROPS_PER_IMAGE)):
            idx = sort_idx[det_i]
            x1, y1, x2, y2 = xyxy[idx].astype(int)
            conf = float(confs[idx])

            w, h = x2 - x1, y2 - y1
            if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                continue

            # Add padding
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(img.width, x2 + PADDING)
            y2 = min(img.height, y2 + PADDING)

            # Extract and save crop
            crop = img.crop((x1, y1, x2, y2))
            fname = f"crop_{crop_count:06d}.jpg"
            crop.save(os.path.join(CROPS_DIR, fname), "JPEG", quality=85)

            metadata.append({
                "crop_id": fname,
                "source_image": source,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 3),
                "detector": "YOLO26s",
            })
            crop_count += 1

    except Exception as e:
        if img_idx < 5:
            print(f"    [WARN] {os.path.basename(img_path)}: {e}")
        continue

# Save metadata
meta_path = os.path.join(WORKING_DIR, "crop_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f)

print(f"\n  [OK] {crop_count} product crops extracted → {CROPS_DIR}")
print(f"  [OK] Metadata saved: {meta_path}")

# Free YOLO from GPU
del yolo_model
torch.cuda.empty_cache()
gc.collect()
print(f"  [OK] YOLO model freed from GPU")

# FREE DISK: Delete SKU-110K dataset (~13 GB) — crops are already extracted
print(f"  Deleting SKU-110K images to free disk space...")
shutil.rmtree(SKU_DIR, ignore_errors=True)
shutil.rmtree("/kaggle/working/datasets", ignore_errors=True)
os.system("df -h /kaggle/working | tail -1")
print(f"  [OK] Disk space freed")


# =============================================================================
# CELL 4: SimCLR Augmentation + Dataset
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2: BUILDING CONTRASTIVE DATASET")
print("=" * 60)

crop_files = sorted(glob.glob(os.path.join(CROPS_DIR, "*.jpg")))
print(f"  Found {len(crop_files)} crops")


class GaussianBlur:
    """Gaussian blur augmentation for SimCLR."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRAugmentation:
    """
    SimCLR-style augmentation: generates two differently-augmented
    views of the same image (positive pair).
    """
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class ProductCropDataset(Dataset):
    """Dataset of product crops with SimCLR augmentation."""

    def __init__(self, crop_files, transform, max_samples=None):
        self.files = crop_files
        if max_samples and len(self.files) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.files), max_samples, replace=False)
            self.files = [self.files[i] for i in sorted(indices)]
        self.transform = transform
        print(f"  Dataset: {len(self.files)} crops")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        view1, view2 = self.transform(img)
        return view1, view2


# =============================================================================
# CELL 5: DINOv2 + SimCLR Model
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3: BUILDING DINOv2 + SimCLR MODEL")
print("=" * 60)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class DINOv2SimCLR(nn.Module):
    """DINOv2 ViT-B/14 backbone with SimCLR projection head."""

    def __init__(self, backbone_name="dinov2_vitb14", proj_dim=256):
        super().__init__()

        print(f"  Loading {backbone_name} from torch.hub...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name,
            pretrained=True
        )
        self.embed_dim = self.backbone.embed_dim  # 768 for ViT-B
        print(f"  Backbone: {backbone_name}, embed_dim={self.embed_dim}")

        # Freeze early layers, fine-tune later layers
        n_blocks = len(self.backbone.blocks)
        freeze_until = n_blocks // 2  # Freeze first 6 of 12 blocks

        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        for i, block in enumerate(self.backbone.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Backbone: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

        self.projector = ProjectionHead(self.embed_dim, 2048, proj_dim)
        proj_params = sum(p.numel() for p in self.projector.parameters())
        print(f"  Projector: {proj_params/1e6:.1f}M params")

    def forward(self, x):
        features = self.backbone(x)       # [B, 768]
        projections = self.projector(features)  # [B, 256]
        return projections

    def get_features(self, x):
        with torch.no_grad():
            return self.backbone(x)


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent (InfoNCE) contrastive loss — core of SimCLR.
    Computed in float32 to avoid half-precision overflow.
    """
    batch_size = z1.shape[0]

    # Force float32 (avoids masked_fill overflow with fp16)
    z = torch.cat([z1, z2], dim=0).float()

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature

    # Positive pair labels: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ], dim=0).to(z.device)

    # Mask self-similarities
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    return F.cross_entropy(sim, labels)


# =============================================================================
# CELL 6: Training
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4: DINOv2 CONTRASTIVE FINE-TUNING")
print("=" * 60)

# --- Hyperparameters (auto-scaled to GPU) ---
IMG_SIZE = 224
TEMPERATURE = 0.07
LR = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
MAX_CROPS = 100000

if vram_gb >= 40:       # A100
    BATCH_SIZE = 256
    EPOCHS = 20
elif vram_gb >= 22:     # L4 / V100
    BATCH_SIZE = 128
    EPOCHS = 15
elif vram_gb >= 14:     # T4
    BATCH_SIZE = 64
    EPOCHS = 15
else:                   # P100
    BATCH_SIZE = 48
    EPOCHS = 12

print(f"  Image size:    {IMG_SIZE}×{IMG_SIZE}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  Epochs:        {EPOCHS}")
print(f"  Temperature:   {TEMPERATURE}")
print(f"  LR:            {LR}")
print(f"  Max crops:     {MAX_CROPS}")

# --- Dataset ---
augmentation = SimCLRAugmentation(img_size=IMG_SIZE)
dataset = ProductCropDataset(crop_files, augmentation, max_samples=MAX_CROPS)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,        # Avoid multiprocessing crash on Kaggle
    pin_memory=True,
    drop_last=True,
)
print(f"  Batches/epoch: {len(dataloader)}")

# --- Model ---
model = DINOv2SimCLR(backbone_name="dinov2_vitb14", proj_dim=256)
model = model.cuda()

# Mixed precision
scaler = torch.amp.GradScaler("cuda")

# Optimizer + cosine schedule with warmup
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

total_steps = EPOCHS * len(dataloader)
warmup_steps = WARMUP_EPOCHS * len(dataloader)

def lr_schedule(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

# --- Training loop ---
print(f"\n  Training started: {datetime.datetime.now()}")
print(f"  Total steps: {total_steps}\n")

best_loss = float("inf")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{EPOCHS}", ncols=100)
    for view1, view2 in pbar:
        view1 = view1.cuda(non_blocking=True)
        view2 = view2.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward in mixed precision
        with torch.amp.autocast("cuda"):
            z1 = model(view1)
            z2 = model(view2)

        # Loss in float32 (outside autocast to prevent overflow)
        loss = nt_xent_loss(z1.float(), z2.float(), temperature=TEMPERATURE)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = epoch_loss / max(n_batches, 1)
    loss_history.append(avg_loss)
    current_lr = scheduler.get_last_lr()[0]

    print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}, lr: {current_lr:.2e}")

    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss

        backbone_path = os.path.join(WORKING_DIR, "dinov2_shelf_finetuned.pth")
        projector_path = os.path.join(WORKING_DIR, "dinov2_projector.pth")

        torch.save(model.backbone.state_dict(), backbone_path)
        torch.save(model.projector.state_dict(), projector_path)
        print(f"  ★ BEST model saved (loss={best_loss:.4f})")

print(f"\n[OK] Training complete! Best loss: {best_loss:.4f}")
print(f"  Finished: {datetime.datetime.now()}")


# =============================================================================
# CELL 7: Package & Summary
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5: PACKAGING")
print("=" * 60)

backbone_path = os.path.join(WORKING_DIR, "dinov2_shelf_finetuned.pth")
projector_path = os.path.join(WORKING_DIR, "dinov2_projector.pth")

# Training info
train_info = {
    "model": "DINOv2 ViT-B/14 (SimCLR fine-tuned)",
    "detector": "YOLO26s (yolo_shelf_best.pt)",
    "dataset": "SKU-110K product crops",
    "n_crops_total": crop_count,
    "n_crops_trained": len(dataset),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "img_size": IMG_SIZE,
    "temperature": TEMPERATURE,
    "lr": LR,
    "best_loss": round(best_loss, 4),
    "loss_history": [round(l, 4) for l in loss_history],
    "embed_dim": 768,
    "proj_dim": 256,
    "gpu": gpu_name,
    "pytorch": torch.__version__,
    "backbone_size_MB": round(os.path.getsize(backbone_path) / 1e6, 1) if os.path.exists(backbone_path) else 0,
}

info_path = os.path.join(WORKING_DIR, "dinov2_training_info.json")
with open(info_path, "w") as f:
    json.dump(train_info, f, indent=2)

# Create zip
zip_path = os.path.join(WORKING_DIR, "dinov2_shelf.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(backbone_path):
        zf.write(backbone_path, "dinov2_shelf_finetuned.pth")
    if os.path.exists(projector_path):
        zf.write(projector_path, "dinov2_projector.pth")
    zf.write(info_path, "dinov2_training_info.json")
    zf.write(meta_path, "crop_metadata.json")

zs = os.path.getsize(zip_path) / 1e6
bs = os.path.getsize(backbone_path) / 1e6 if os.path.exists(backbone_path) else 0

print(f"  [OK] dinov2_shelf_finetuned.pth  ({bs:.1f} MB)")
print(f"  [OK] dinov2_projector.pth")
print(f"  [OK] dinov2_shelf.zip            ({zs:.1f} MB)")
print(f"  [OK] dinov2_training_info.json")
print(f"  [OK] crop_metadata.json          ({crop_count} crops)")

# Cleanup
del model
torch.cuda.empty_cache()
gc.collect()


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print("  ★ SHELFMIND AI — PIPELINE COMPLETE ★")
print(f"{'='*60}")
print(f"""
  YOLO26s Crop Extraction:
    - Model:    yolo_shelf_best.pt
    - Images:   {len(all_images)}
    - Crops:    {crop_count}

  DINOv2 Contrastive Fine-Tuning:
    - Backbone: DINOv2 ViT-B/14
    - Method:   SimCLR (NT-Xent loss)
    - Crops:    {len(dataset)}
    - Epochs:   {EPOCHS}
    - Best loss: {best_loss:.4f}
    - Embed dim: 768

  Files in /kaggle/working/:
    - crops_for_dinov2/           (product crops)
    - dinov2_shelf_finetuned.pth  (fine-tuned backbone)
    - dinov2_projector.pth        (projection head)
    - dinov2_shelf.zip            (packaged models)
    - crop_metadata.json          (bbox metadata)
    - dinov2_training_info.json   (training config)

  USAGE in production:
    ```python
    import torch
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    backbone.load_state_dict(torch.load('dinov2_shelf_finetuned.pth'))
    backbone.eval().cuda()

    # Get product embedding
    features = backbone(image_tensor)  # [B, 768]
    ```

  ShelfMind Pipeline:
    YOLO26s → detect products → DINOv2 → product embeddings → match/identify
""")
