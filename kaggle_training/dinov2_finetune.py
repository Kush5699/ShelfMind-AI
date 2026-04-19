"""
=============================================================================
ShelfMind AI — DINOv2 Contrastive Fine-Tuning on Product Crops
=============================================================================
Fine-tunes Meta's DINOv2 ViT-B/14 backbone using SimCLR-style self-supervised
contrastive learning on product crops extracted by RF-DETR from SKU-110K.

Goal: Learn discriminative product embeddings for shelf product matching.

SETUP (Lightning.ai):
1. Upload extracted_crops.zip (from RF-DETR step)
2. Upload this script
3. Run: python dinov2_finetune.py

OUTPUT:
  - dinov2_shelf_finetuned.pth  (~330 MB) → Fine-tuned DINOv2 backbone
  - dinov2_projector.pth        (~2 MB)   → Projection head
  - dinov2_shelf.zip            (packed)  → Ready for deployment
=============================================================================
"""

# =============================================================================
# STEP 0: Install dependencies
# =============================================================================
import subprocess
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(args))

run_pip("timm", "pillow", "tqdm")
print("[OK] Dependencies installed")

# =============================================================================
# Imports
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
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Config
# =============================================================================
OUTPUT_DIR = "shelfmind_output"
CROPS_DIR = None  # Auto-detected below
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging ---
class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", encoding="utf-8", buffering=1)
        self.log.write(f"=== DINOv2 Fine-Tuning Log ===\n")
        self.log.write(f"Started: {datetime.datetime.now()}\n\n")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(os.path.join(OUTPUT_DIR, "dinov2_log.txt"))

# --- GPU ---
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU required!"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)\n")


# =============================================================================
# STEP 1: Locate / Extract Crops
# =============================================================================
print("=" * 60)
print("  STEP 1: LOCATE PRODUCT CROPS")
print("=" * 60)

# Check multiple possible locations
candidates = [
    "shelfmind_output/crops_for_dinov2",
    "crops_for_dinov2",
    "extracted_crops/crops",
    "crops",
]

for c in candidates:
    if os.path.exists(c) and len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
        CROPS_DIR = c
        break

# Try extracting from zip if not found
if CROPS_DIR is None:
    zip_candidates = [
        "shelfmind_output/extracted_crops.zip",
        "extracted_crops.zip",
    ]
    for zc in zip_candidates:
        if os.path.exists(zc):
            print(f"  Extracting {zc}...")
            with zipfile.ZipFile(zc, "r") as z:
                z.extractall("extracted_crops_data")
            # Find the crops dir inside
            for root, dirs, files in os.walk("extracted_crops_data"):
                jpgs = [f for f in files if f.endswith(".jpg")]
                if len(jpgs) > 100:
                    CROPS_DIR = root
                    break
            if CROPS_DIR:
                break

assert CROPS_DIR is not None, "No crops found! Upload extracted_crops.zip first."

crop_files = sorted(glob.glob(os.path.join(CROPS_DIR, "*.jpg")))
print(f"  [OK] Found {len(crop_files)} product crops in {CROPS_DIR}")

# Load metadata if available
meta_path = None
for mp in [
    os.path.join(OUTPUT_DIR, "crop_metadata.json"),
    os.path.join(os.path.dirname(CROPS_DIR), "crop_metadata.json"),
    "crop_metadata.json",
]:
    if os.path.exists(mp):
        meta_path = mp
        break

if meta_path:
    with open(meta_path) as f:
        crop_metadata = json.load(f)
    print(f"  [OK] Metadata loaded: {len(crop_metadata)} entries")
else:
    crop_metadata = None
    print("  [INFO] No metadata file found (that's OK)")


# =============================================================================
# STEP 2: SimCLR Augmentation + Dataset
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2: BUILDING CONTRASTIVE DATASET")
print("=" * 60)

class GaussianBlur:
    """Gaussian blur augmentation for SimCLR."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRAugmentation:
    """
    SimCLR-style augmentation pipeline for product crops.
    Generates two differently-augmented views of the same image.
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
            # Random subsample for very large datasets
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
            # Return a blank image if file is corrupt
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        view1, view2 = self.transform(img)
        return view1, view2


# =============================================================================
# STEP 3: DINOv2 Backbone + Projection Head
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 3: BUILDING DINOv2 + SimCLR MODEL")
print("=" * 60)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (SimCLR-style)."""
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
    """DINOv2 backbone with SimCLR projection head."""

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
        # DINOv2 ViT-B has 12 transformer blocks
        n_blocks = len(self.backbone.blocks)
        freeze_until = n_blocks // 2  # Freeze first 6 blocks

        # Freeze patch embed + early blocks
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
        """Returns normalized projection embeddings."""
        features = self.backbone(x)  # [B, 768]
        projections = self.projector(features)  # [B, 256]
        return projections

    def get_features(self, x):
        """Returns raw backbone features (for downstream use)."""
        with torch.no_grad():
            return self.backbone(x)


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
    The core SimCLR contrastive loss function.
    """
    batch_size = z1.shape[0]

    # Cast to float32 to avoid half-precision overflow with masked_fill
    z = torch.cat([z1, z2], dim=0).float()  # [2B, D]

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Create labels: positive pairs are (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ], dim=0).to(z.device)

    # Mask out self-similarities (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Cross-entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss


# =============================================================================
# STEP 4: Training Loop
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4: CONTRASTIVE FINE-TUNING")
print("=" * 60)

# --- Hyperparameters ---
IMG_SIZE = 224          # DINOv2 ViT-B/14: 224px (14px patches → 16×16 tokens)
TEMPERATURE = 0.07      # NT-Xent temperature (lower = harder negatives)
LR = 5e-5               # Learning rate for fine-tuning (low for pretrained)
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
MAX_CROPS = 100000      # Cap dataset size for reasonable training time

if vram_gb >= 40:       # A100
    BATCH_SIZE = 256
    EPOCHS = 20
elif vram_gb >= 22:     # L4
    BATCH_SIZE = 128
    EPOCHS = 15
else:
    BATCH_SIZE = 64
    EPOCHS = 10

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
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
print(f"  Batches/epoch: {len(dataloader)}")

# --- Model ---
model = DINOv2SimCLR(backbone_name="dinov2_vitb14", proj_dim=256)
model = model.cuda()

# Use mixed precision for speed
scaler = torch.amp.GradScaler("cuda")

# --- Optimizer with cosine schedule + warmup ---
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

# --- Training ---
print(f"\n  Starting training ({EPOCHS} epochs, {total_steps} steps)...\n")

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

        with torch.amp.autocast("cuda"):
            z1 = model(view1)
            z2 = model(view2)

        # Compute loss in float32 (outside autocast to avoid half-precision overflow)
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

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.backbone.state_dict(),
                   os.path.join(OUTPUT_DIR, "dinov2_shelf_finetuned.pth"))
        torch.save(model.projector.state_dict(),
                   os.path.join(OUTPUT_DIR, "dinov2_projector.pth"))
        print(f"  [BEST] Saved (loss={best_loss:.4f})")

print(f"\n[OK] Training complete! Best loss: {best_loss:.4f}")


# =============================================================================
# STEP 5: Save + Package
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5: PACKAGING")
print("=" * 60)

backbone_path = os.path.join(OUTPUT_DIR, "dinov2_shelf_finetuned.pth")
projector_path = os.path.join(OUTPUT_DIR, "dinov2_projector.pth")

# Save training info
train_info = {
    "model": "DINOv2 ViT-B/14 (SimCLR fine-tuned)",
    "dataset": "SKU-110K product crops (RF-DETR extracted)",
    "n_crops": len(dataset),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "img_size": IMG_SIZE,
    "temperature": TEMPERATURE,
    "lr": LR,
    "best_loss": best_loss,
    "loss_history": loss_history,
    "embed_dim": 768,
    "proj_dim": 256,
    "backbone_size_MB": os.path.getsize(backbone_path) / 1e6 if os.path.exists(backbone_path) else 0,
}

info_path = os.path.join(OUTPUT_DIR, "dinov2_training_info.json")
with open(info_path, "w") as f:
    json.dump(train_info, f, indent=2)

# Package
zip_path = os.path.join(OUTPUT_DIR, "dinov2_shelf.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    if os.path.exists(backbone_path):
        zf.write(backbone_path, "dinov2_shelf_finetuned.pth")
    if os.path.exists(projector_path):
        zf.write(projector_path, "dinov2_projector.pth")
    zf.write(info_path, "dinov2_training_info.json")

zs = os.path.getsize(zip_path) / 1e6
bs = os.path.getsize(backbone_path) / 1e6 if os.path.exists(backbone_path) else 0

print(f"  [OK] dinov2_shelf_finetuned.pth  ({bs:.1f} MB)")
print(f"  [OK] dinov2_shelf.zip            ({zs:.1f} MB)")
print(f"  [OK] dinov2_training_info.json")

# Cleanup
del model
torch.cuda.empty_cache()
gc.collect()


# =============================================================================
# DONE
# =============================================================================
print(f"\n{'='*60}")
print("  SHELFMIND AI — DINOv2 FINE-TUNING COMPLETE!")
print(f"{'='*60}")
print(f"\n  Backbone:   DINOv2 ViT-B/14")
print(f"  Method:     SimCLR contrastive learning")
print(f"  Crops:      {len(dataset)}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Best loss:  {best_loss:.4f}")
print(f"  Embed dim:  768")
print(f"\n  USAGE in production:")
print(f"  ```python")
print(f"  backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')")
print(f"  backbone.load_state_dict(torch.load('dinov2_shelf_finetuned.pth'))")
print(f"  features = backbone(image_tensor)  # [B, 768]")
print(f"  ```")
print(f"\n  Download from shelfmind_output/:")
print(f"  - dinov2_shelf.zip")
print(f"\nFinished: {datetime.datetime.now()}")
