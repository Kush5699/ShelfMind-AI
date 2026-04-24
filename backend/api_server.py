"""
=============================================================================
ShelfMind AI — FastAPI Backend (HuggingFace Spaces Docker Deployment)
=============================================================================
Production-ready API serving YOLO26s detection, DINOv2 embeddings,
FAISS product matching, rembg auto-crop, and EasyOCR.

Run locally:  python api_server.py
HF Spaces:    Auto-starts via Dockerfile CMD
=============================================================================
"""

import os
import sys
import io
import gc
import re
import json
import time
import base64
import hashlib
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F

# ── FastAPI ───────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CATALOG_DIR = DATA_DIR / "store_catalog"
REF_IMG_DIR = CATALOG_DIR / "reference_images"
PLANOGRAM_DIR = DATA_DIR / "store_planograms"

for d in [DATA_DIR, CATALOG_DIR, REF_IMG_DIR, PLANOGRAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shelfmind")

# =============================================================================
# MODEL MANAGER (Singleton — loaded once at startup)
# =============================================================================
class ModelManager:
    """Manages all ML models. Loads once, serves forever."""

    def __init__(self):
        self.yolo = None
        self.dinov2 = None
        self.projector = None
        self.ocr_reader = None
        self.rembg_session = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False

    def load_all(self):
        """Load all models at startup."""
        start = time.time()
        logger.info(f"Loading models on device={self.device}...")

        self._load_yolo()
        self._load_dinov2()
        self._load_ocr()
        self._load_rembg()

        self.ready = True
        elapsed = time.time() - start
        logger.info(f"[OK] All models loaded in {elapsed:.1f}s")

    def _load_yolo(self):
        yolo_path = MODEL_DIR / "yolo_shelf_best.pt"
        if yolo_path.exists():
            from ultralytics import YOLO
            self.yolo = YOLO(str(yolo_path))
            logger.info(f"  [OK] YOLO26s: {yolo_path.stat().st_size/1e6:.1f} MB")
        else:
            logger.warning(f"  [SKIP] YOLO not found at {yolo_path}")

    def _load_dinov2(self):
        finetuned_path = MODEL_DIR / "dinov2_shelf_finetuned.pth"
        projector_path = MODEL_DIR / "dinov2_projector.pth"

        try:
            # Load base DINOv2 from torch.hub (cached from Docker build)
            self.dinov2 = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=True, verbose=False
            )
            # Load our fine-tuned weights on top
            if finetuned_path.exists():
                state = torch.load(str(finetuned_path), map_location=self.device, weights_only=True)
                self.dinov2.load_state_dict(state, strict=False)
                logger.info(f"  [OK] DINOv2: fine-tuned weights loaded ({finetuned_path.stat().st_size/1e6:.1f} MB)")
            else:
                logger.info("  [OK] DINOv2: using pretrained base (no fine-tuned weights)")
            self.dinov2.eval().to(self.device)

            # Load projector head if available
            if projector_path.exists():
                self.projector = torch.load(str(projector_path), map_location=self.device, weights_only=True)
                logger.info(f"  [OK] DINOv2 projector loaded ({projector_path.stat().st_size/1e6:.1f} MB)")

        except Exception as e:
            logger.warning(f"  [SKIP] DINOv2 load error: {e}")
            self.dinov2 = None

    def _load_ocr(self):
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))
            logger.info("  [OK] EasyOCR loaded")
        except Exception as e:
            logger.warning(f"  [SKIP] EasyOCR error: {e}")

    def _load_rembg(self):
        try:
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            logger.info("  [OK] rembg (U2-Net) loaded")
        except Exception as e:
            logger.warning(f"  [SKIP] rembg error: {e}")

    # ── Inference Methods ─────────────────────────────────

    def detect(self, image: Image.Image, conf: float = 0.3, max_det: int = 200) -> list:
        """Run YOLO26s detection. Returns list of detections."""
        if not self.yolo:
            raise RuntimeError("YOLO model not loaded")

        results = self.yolo.predict(image, conf=conf, max_det=max_det, device="cpu", verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(float(box.conf[0]), 3),
                    "class": int(box.cls[0]) if box.cls is not None else 0,
                })
        return detections

    def get_embedding(self, crop: Image.Image) -> np.ndarray:
        """Get DINOv2 embedding for a product crop. Returns L2-normalized 768-dim vector."""
        if not self.dinov2:
            return np.zeros(768, dtype=np.float32)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.dinov2(tensor).squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def get_robust_embedding(self, image: Image.Image):
        """Generate robust embedding by averaging 10 augmented views.
        Returns (embedding, view_names).
        """
        img = image.convert("RGB")
        w, h = img.size

        views = []
        view_names = []

        # 1. Original
        views.append(img)
        view_names.append("Original")

        # 2. Horizontal flip
        views.append(ImageOps.mirror(img))
        view_names.append("H-Flip")

        # 3. Center crop 80%
        mw, mh = int(w * 0.1), int(h * 0.1)
        views.append(img.crop((mw, mh, w - mw, h - mh)))
        view_names.append("Center 80%")

        # 4. Left crop
        views.append(img.crop((0, mh, int(w * 0.85), h - mh)))
        view_names.append("Left Crop")

        # 5. Right crop
        views.append(img.crop((int(w * 0.15), mh, w, h - mh)))
        view_names.append("Right Crop")

        # 6. Top crop
        views.append(img.crop((mw, 0, w - mw, int(h * 0.85))))
        view_names.append("Top Crop")

        # 7. Rotation +5°
        views.append(img.rotate(-5, expand=False, fillcolor=(128, 128, 128)))
        view_names.append("Rotate +5°")

        # 8. Rotation -5°
        views.append(img.rotate(5, expand=False, fillcolor=(128, 128, 128)))
        view_names.append("Rotate -5°")

        # 9. Brightness +20%
        views.append(ImageEnhance.Brightness(img).enhance(1.2))
        view_names.append("Bright +20%")

        # 10. Contrast +20%
        views.append(ImageEnhance.Contrast(img).enhance(1.2))
        view_names.append("Contrast +20%")

        # Average all embeddings
        embeddings = [self.get_embedding(v) for v in views]
        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        return avg_emb, view_names

    def auto_crop_rembg(self, image: Image.Image) -> Image.Image:
        """Remove background with rembg and crop to product bounds."""
        if not self.rembg_session:
            return image

        from rembg import remove
        result = remove(image, session=self.rembg_session)
        alpha = np.array(result.split()[-1])
        coords = np.where(alpha > 30)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            pad = 5
            h, w = alpha.shape
            x1 = max(0, x_min - pad)
            y1 = max(0, y_min - pad)
            x2 = min(w, x_max + pad)
            y2 = min(h, y_max + pad)
            return image.crop((x1, y1, x2, y2))
        return image

    def read_ocr(self, image: Image.Image) -> str:
        """Extract text from product image using EasyOCR."""
        if not self.ocr_reader:
            return ""
        try:
            img_np = np.array(image)
            results = self.ocr_reader.readtext(img_np, detail=0, paragraph=True)
            return " ".join(results).strip()
        except Exception:
            return ""

    def scan_barcode(self, image: Image.Image) -> str:
        """Detect barcode from image using pyzbar with multiple rotations."""
        try:
            from pyzbar.pyzbar import decode
            import cv2
            img_cv = np.array(image)

            # Try 4 rotations × 2 preprocessing
            for angle in [0, 90, 180, 270]:
                for preprocess in [False, True]:
                    rotated = img_cv
                    if angle > 0:
                        if angle == 90:
                            rotated = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 180:
                            rotated = cv2.rotate(img_cv, cv2.ROTATE_180)
                        elif angle == 270:
                            rotated = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    if preprocess:
                        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
                        rotated = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

                    barcodes = decode(rotated)
                    if barcodes:
                        return barcodes[0].data.decode("utf-8")
            return ""
        except Exception:
            return ""

    def lookup_barcode(self, barcode: str) -> dict:
        """Look up product info from Open Food Facts API."""
        try:
            import requests
            resp = requests.get(
                f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == 1:
                    product = data.get("product", {})
                    return {
                        "name": product.get("product_name", ""),
                        "category": product.get("categories_tags", [""])[0].replace("en:", "").replace("-", " ").title() if product.get("categories_tags") else "",
                        "brand": product.get("brands", ""),
                    }
        except Exception:
            pass
        return {}

    def detect_shelf_levels(self, detections: list, img_height: int):
        """Auto-detect shelf levels from product Y-positions."""
        if len(detections) < 3:
            return [0.0, float(img_height)], 1

        y_centers = sorted([(d["bbox"][1] + d["bbox"][3]) / 2 for d in detections])
        gaps = [(y_centers[i] - y_centers[i-1], i) for i in range(1, len(y_centers))]

        min_shelf_gap = img_height * 0.15
        small_gaps = sorted([g for g, _ in gaps])[:max(len(gaps)//2, 1)]
        avg_small = sum(small_gaps) / len(small_gaps) if small_gaps else 10
        stat_threshold = avg_small * 3
        sig_threshold = max(min_shelf_gap, stat_threshold)

        gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)
        top_gaps = [(g, idx) for g, idx in gaps_sorted if g > sig_threshold][:6]
        top_gaps_sorted = sorted(top_gaps, key=lambda x: y_centers[x[1]])

        boundaries = [0.0]
        for _, idx in top_gaps_sorted:
            boundaries.append((y_centers[idx-1] + y_centers[idx]) / 2)
        boundaries.append(float(img_height))

        return boundaries, len(boundaries) - 1

    def assign_to_shelves(self, detections: list, boundaries: list) -> dict:
        """Assign each detection to a shelf level."""
        shelf_assignments = {}
        for det in detections:
            y_center = (det["bbox"][1] + det["bbox"][3]) / 2
            for s in range(len(boundaries) - 1):
                if boundaries[s] <= y_center < boundaries[s + 1]:
                    shelf_id = s + 1
                    if shelf_id not in shelf_assignments:
                        shelf_assignments[shelf_id] = []
                    det["shelf"] = shelf_id
                    shelf_assignments[shelf_id].append(det)
                    break
        for shelf_id in shelf_assignments:
            shelf_assignments[shelf_id].sort(key=lambda d: d["bbox"][0])
        return shelf_assignments

    def search_product_faiss(self, query_embedding, products, threshold=0.3,
                              query_bbox=None, expected_products=None, size_weight=0.15):
        """Match product embedding against catalog using FAISS-style cosine similarity with size-ratio fusion."""
        if not products:
            return None, 0.0

        # Build embeddings matrix from catalog
        valid_products = [p for p in products if p.get("embedding")]
        if not valid_products:
            return None, 0.0

        catalog_embeddings = np.array([p["embedding"] for p in valid_products], dtype=np.float32)
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Cosine similarity
        norms_cat = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
        norms_cat[norms_cat == 0] = 1
        catalog_norm = catalog_embeddings / norms_cat

        norm_q = np.linalg.norm(query)
        if norm_q > 0:
            query_norm = query / norm_q
        else:
            query_norm = query

        scores = (catalog_norm @ query_norm.T).flatten()

        # Get top-3 candidates
        k = min(3, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        if scores[top_indices[0]] < threshold:
            return None, float(scores[top_indices[0]])

        if expected_products is None or query_bbox is None:
            best_idx = top_indices[0]
            return valid_products[best_idx], float(scores[best_idx])

        # Size-ratio fusion
        qx1, qy1, qx2, qy2 = query_bbox
        query_height = qy2 - qy1

        best_match = None
        best_combined = 0.0

        for idx in top_indices:
            visual_score = float(scores[idx])
            if visual_score < threshold * 0.8:
                continue
            candidate = valid_products[idx]
            candidate_sku = candidate["sku"]

            size_score = 1.0
            for exp_prod in expected_products:
                if exp_prod.get("sku") == candidate_sku:
                    exp_bbox = exp_prod.get("bbox")
                    if exp_bbox and len(exp_bbox) == 4:
                        exp_height = exp_bbox[3] - exp_bbox[1]
                        if exp_height > 0 and query_height > 0:
                            size_score = min(query_height, exp_height) / max(query_height, exp_height)
                    break

            combined = visual_score * (1 - size_weight) + size_score * size_weight
            if combined > best_combined:
                best_combined = combined
                best_match = candidate

        if best_match and best_combined >= threshold:
            return best_match, best_combined
        return valid_products[top_indices[0]], float(scores[top_indices[0]])

    def draw_annotated(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes with color-coded labels."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det.get("match_score", det.get("confidence", 0))

            if det.get("status") == "unknown" or score < 0.3:
                color = "#ff4343"
            elif score >= 0.7:
                color = "#00d4aa"
            elif score >= 0.5:
                color = "#ffaa00"
            else:
                color = "#ff8c00"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = det.get("product_name", f"P{det.get('confidence', 0):.0%}")
            text = f"{label} ({score:.0%})"
            bbox = draw.textbbox((x1, y1 - 18), text, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            draw.text((x1, y1 - 18), text, fill="#0a0a1a", font=font)

        return img


# =============================================================================
# DATABASE LAYER
# =============================================================================
sys.path.insert(0, str(ROOT))
try:
    from db import (
        setup_database, add_product, get_products, get_product_count,
        get_next_product_id, delete_product, clear_all_products,
        get_catalog_as_dict, save_planogram_db, get_planograms,
        delete_planogram, log_compliance, log_alert,
        get_compliance_logs_as_list, get_analytics_summary, get_alerts_history,
    )
    setup_database()
    logger.info("[OK] Database initialized")
except ImportError as e:
    logger.warning(f"[SKIP] db.py not found: {e}")
    # Minimal fallback
    _products = []
    def get_products(): return _products
    def add_product(**kwargs): _products.append(kwargs); return kwargs.get("sku")
    def get_product_count(): return len(_products)
    def get_next_product_id(): return len(_products) + 1
    def delete_product(sku): pass
    def clear_all_products(): _products.clear()
    def get_catalog_as_dict(): return {"products": _products, "next_id": len(_products)+1}
    def get_planograms(): return {}
    def save_planogram_db(name, data): pass
    def delete_planogram(name): pass
    def get_compliance_logs_as_list(): return []
    def get_analytics_summary(): return {}
    def get_alerts_history(): return []
    def log_compliance(**kwargs): return 0
    def log_alert(**kwargs): pass

# =============================================================================
# FASTAPI APP
# =============================================================================
models = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_all()
    yield
    del models.yolo, models.dinov2
    torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(
    title="ShelfMind AI",
    description="Smart Retail Shelf Intelligence API — YOLO26s + DINOv2 + FAISS",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────
def image_to_base64(img: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def read_upload_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {"status": "ShelfMind AI Backend is running", "version": "2.0.0"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy" if models.ready else "loading",
        "device": models.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models": {
            "yolo": models.yolo is not None,
            "dinov2": models.dinov2 is not None,
            "ocr": models.ocr_reader is not None,
            "rembg": models.rembg_session is not None,
        },
        "products_registered": get_product_count(),
        "timestamp": datetime.now().isoformat(),
    }


# ── Product Scanner: Single Product ──────────────────────

@app.post("/api/scan/single")
async def scan_single_product(image: UploadFile = File(...)):
    """Full single product scan: rembg crop → barcode → OCR → robust embedding."""
    contents = await image.read()
    img = read_upload_image(contents)

    # 1. Barcode scan
    barcode = models.scan_barcode(img)
    barcode_info = {}
    if barcode:
        barcode_info = models.lookup_barcode(barcode)

    # 2. OCR
    ocr_text = models.read_ocr(img)

    # 3. Auto-crop with rembg
    cropped = models.auto_crop_rembg(img)

    # 4. Robust embedding (10 augmented views)
    embedding, view_names = models.get_robust_embedding(cropped)

    return {
        "barcode": barcode,
        "barcode_info": barcode_info,
        "ocr_text": ocr_text,
        "cropped_image": image_to_base64(cropped),
        "original_image": image_to_base64(img),
        "embedding": embedding.tolist(),
        "augmentation_views": view_names,
        "embedding_dim": len(embedding),
    }


# ── Product Scanner: Bulk Shelf Scan ─────────────────────

@app.post("/api/scan/bulk")
async def scan_bulk_shelf(image: UploadFile = File(...), similarity_threshold: float = Form(0.82)):
    """Bulk shelf scan: YOLO detect → DINOv2 cluster unique products → OCR."""
    contents = await image.read()
    img = read_upload_image(contents)

    # Step 1: Detect all products
    detections = models.detect(img, conf=0.3)

    # Step 2: Crop and embed each
    crops_data = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        emb = models.get_embedding(crop)
        crops_data.append({
            "crop": crop,
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "embedding": emb,
        })

    # Step 3: Cluster unique products by cosine similarity
    unique_products = []
    used = set()

    for i, cd in enumerate(crops_data):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(crops_data)):
            if j in used:
                continue
            sim = float(np.dot(cd["embedding"], crops_data[j]["embedding"]))
            if sim >= similarity_threshold:
                cluster.append(j)
                used.add(j)
        used.add(i)

        # OCR on representative crop
        ocr_text = models.read_ocr(cd["crop"])

        unique_products.append({
            "crop_image": image_to_base64(cd["crop"]),
            "bbox": cd["bbox"],
            "confidence": cd["confidence"],
            "embedding": cd["embedding"].tolist(),
            "ocr_text": ocr_text,
            "count": len(cluster),
            "geometry": {
                "width": cd["bbox"][2] - cd["bbox"][0],
                "height": cd["bbox"][3] - cd["bbox"][1],
            },
        })

    # Draw annotated image
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

    return {
        "total_detected": len(detections),
        "unique_products": len(unique_products),
        "products": unique_products,
        "annotated_image": image_to_base64(annotated),
    }


# ── Product CRUD ──────────────────────────────────────────

@app.get("/api/products")
async def list_products():
    """List all registered products."""
    products = get_products()
    result = []
    for p in products:
        prod = dict(p) if hasattr(p, 'keys') else p
        # Convert embedding to flag (don't send full embedding)
        prod["has_embedding"] = prod.get("embedding") is not None and len(prod.get("embedding", [])) > 0
        if "embedding" in prod:
            del prod["embedding"]
        # Add image URL
        if prod.get("image_path"):
            prod["image_url"] = f"/api/products/{prod['sku']}/image"
        result.append(prod)
    return {"products": result, "count": len(result)}


@app.post("/api/products")
async def create_product(
    name: str = Form(...),
    category: str = Form("Other"),
    price: float = Form(0.0),
    barcode: str = Form(""),
    embedding: str = Form(""),  # JSON string
    image: Optional[UploadFile] = File(None),
):
    """Register a new product."""
    next_id = get_next_product_id()
    sku = f"SKU_{next_id:04d}"

    # Save image
    img_filename = ""
    if image:
        contents = await image.read()
        img = read_upload_image(contents)
        img_filename = f"{sku}_{re.sub(r'[^a-z0-9_]', '', name.replace(' ', '_').lower())}.jpg"
        img_path = REF_IMG_DIR / img_filename
        img.save(str(img_path), "JPEG", quality=90)

    # Parse embedding
    emb = None
    if embedding:
        try:
            emb = json.loads(embedding)
        except json.JSONDecodeError:
            pass

    add_product(
        sku=sku, name=name, category=category,
        price=price, image_path=img_filename,
        embedding=emb, barcode=barcode if barcode else None,
    )

    return {"status": "ok", "sku": sku, "name": name}


@app.get("/api/products/{sku}/image")
async def get_product_image(sku: str):
    """Serve product reference image."""
    products = get_products()
    for p in products:
        if p["sku"] == sku and p.get("image_path"):
            img_path = REF_IMG_DIR / p["image_path"]
            if img_path.exists():
                return FileResponse(str(img_path), media_type="image/jpeg")
    raise HTTPException(404, "Image not found")


@app.delete("/api/products/{sku}")
async def remove_product(sku: str):
    """Delete a product by SKU."""
    delete_product(sku)
    return {"status": "ok"}


@app.delete("/api/products")
async def clear_products():
    """Delete all products."""
    clear_all_products()
    return {"status": "ok"}


@app.post("/api/products/voice")
async def voice_product(transcript: str = Form(...)):
    """Parse voice transcript into product fields."""
    text = transcript.lower().strip()

    # Extract price
    price = 0.0
    price_match = re.search(r'(?:price|rs|rupees?|₹)\s*(\d+(?:\.\d+)?)', text)
    if not price_match:
        price_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:rupees?|rs)', text)
    if price_match:
        price = float(price_match.group(1))

    # Extract category
    category = "Other"
    categories = {
        "snack": "Snacks", "biscuit": "Snacks", "chip": "Snacks", "namkeen": "Snacks",
        "drink": "Beverages", "juice": "Beverages", "water": "Beverages", "cola": "Beverages",
        "dairy": "Dairy", "milk": "Dairy", "curd": "Dairy", "paneer": "Dairy", "cheese": "Dairy",
        "soap": "Personal Care", "shampoo": "Personal Care", "cream": "Personal Care",
        "detergent": "Household", "cleaner": "Household", "wash": "Household",
        "rice": "Staples", "flour": "Staples", "atta": "Staples", "dal": "Staples",
        "chocolate": "Confectionery", "candy": "Confectionery",
        "spice": "Spices", "masala": "Spices",
    }
    for keyword, cat in categories.items():
        if keyword in text:
            category = cat
            break

    # Name = transcript minus price/category mentions
    name_text = text
    for pattern in [r'(?:price|rs|rupees?|₹)\s*\d+(?:\.\d+)?', r'\d+(?:\.\d+)?\s*(?:rupees?|rs)', r'category\s+\w+']:
        name_text = re.sub(pattern, '', name_text, flags=re.I)
    name = name_text.strip().strip(',').strip().title() or "Unknown Product"

    return {"parsed": {"name": name, "category": category, "price": price}}


# ── Planogram CRUD ────────────────────────────────────────

@app.post("/api/planogram/auto-detect")
async def auto_detect_planogram(image: UploadFile = File(...), confidence: float = Form(0.25)):
    """Upload shelf image → detect products → identify → return planogram data."""
    contents = await image.read()
    img = read_upload_image(contents)

    # Detect
    detections = models.detect(img, conf=confidence)
    boundaries, n_shelves = models.detect_shelf_levels(detections, img.height)
    shelf_assignments = models.assign_to_shelves(detections, boundaries)

    # Identify each product
    catalog_products = get_products()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        emb = models.get_embedding(crop)
        match, score = models.search_product_faiss(
            emb, catalog_products, threshold=0.3,
            query_bbox=(x1, y1, x2, y2),
        )
        if match:
            det["product_name"] = match["name"]
            det["product_sku"] = match["sku"]
            det["match_score"] = round(score, 3)
        else:
            det["product_name"] = "Unknown"
            det["product_sku"] = "UNKNOWN"
            det["match_score"] = round(score, 3)

    # Build planogram structure
    shelves = []
    for shelf_id in sorted(shelf_assignments.keys()):
        shelf_dets = shelf_assignments[shelf_id]
        products_on_shelf = []
        for pos, det in enumerate(shelf_dets):
            products_on_shelf.append({
                "position": pos,
                "sku": det.get("product_sku", "UNKNOWN"),
                "name": det.get("product_name", "Unknown"),
                "confidence": det.get("match_score", 0),
                "bbox": det["bbox"],
            })
        shelves.append({"level": shelf_id, "product_count": len(shelf_dets), "products": products_on_shelf})

    # Annotated image
    annotated = models.draw_annotated(img, detections)

    return {
        "n_shelves": n_shelves,
        "total_products": len(detections),
        "shelves": shelves,
        "annotated_image": image_to_base64(annotated),
    }


@app.post("/api/planograms")
async def save_planogram(
    name: str = Form(...),
    data: str = Form(...),  # JSON string of planogram data
    image: Optional[UploadFile] = File(None),
):
    """Save a planogram."""
    planogram_data = json.loads(data)
    planogram_data["name"] = name
    planogram_data["created_at"] = datetime.now().isoformat()
    save_planogram_db(name, planogram_data)

    # Save reference image
    if image:
        contents = await image.read()
        img = read_upload_image(contents)
        img.save(str(PLANOGRAM_DIR / f"{name}_reference.jpg"), "JPEG", quality=90)

    return {"status": "ok", "name": name}


@app.get("/api/planograms")
async def list_planograms():
    """List all planograms."""
    planograms = get_planograms()
    return {"planograms": planograms}


@app.delete("/api/planograms/{name}")
async def remove_planogram(name: str):
    """Delete a planogram."""
    delete_planogram(name)
    ref_img = PLANOGRAM_DIR / f"{name}_reference.jpg"
    if ref_img.exists():
        ref_img.unlink()
    return {"status": "ok"}


# ── Compliance Check ──────────────────────────────────────

@app.post("/api/compliance/check")
async def check_compliance(
    image: UploadFile = File(...),
    planogram_name: str = Form(...),
    confidence: float = Form(0.3),
):
    """Run compliance check: detect → identify → compare vs planogram."""
    contents = await image.read()
    img = read_upload_image(contents)

    # Resize for performance
    max_dim = 640
    if img.width > max_dim:
        ratio = max_dim / img.width
        img = img.resize((max_dim, int(img.height * ratio)), Image.LANCZOS)

    # Get planogram
    planograms = get_planograms()
    if planogram_name not in planograms:
        raise HTTPException(404, f"Planogram '{planogram_name}' not found")
    planogram = planograms[planogram_name]

    # Detect
    detections = models.detect(img, conf=confidence)
    boundaries, n_shelves = models.detect_shelf_levels(detections, img.height)
    shelf_assignments = models.assign_to_shelves(detections, boundaries)

    # Identify products
    catalog_products = get_products()
    all_expected = []
    for ps in planogram.get("shelves", []):
        all_expected.extend(ps.get("products", []))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img.crop((max(0, x1), max(0, y1), x2, y2))
        emb = models.get_embedding(crop)
        match, score = models.search_product_faiss(
            emb, catalog_products, threshold=0.3,
            query_bbox=(x1, y1, x2, y2),
            expected_products=all_expected, size_weight=0.15,
        )
        if match:
            det["product_name"] = match["name"]
            det["product_sku"] = match["sku"]
            det["product_price"] = match.get("price", 0)
            det["match_score"] = round(score, 3)
            det["status"] = "match"
        else:
            det["product_name"] = "Unknown"
            det["product_sku"] = "UNKNOWN"
            det["product_price"] = 0
            det["match_score"] = round(score, 3)
            det["status"] = "unknown"

    # Compliance comparison
    all_alerts = []
    shelf_compliance = {}
    total_expected = 0
    total_matched = 0

    for plan_shelf in planogram.get("shelves", []):
        shelf_id = plan_shelf["level"]
        expected_products = plan_shelf.get("products", [])
        detected_on_shelf = shelf_assignments.get(shelf_id, [])

        expected_counts = Counter(p["sku"] for p in expected_products if p["sku"] != "UNKNOWN")
        detected_counts = Counter(
            d.get("product_sku", "UNKNOWN") for d in detected_on_shelf
            if d.get("product_sku") != "UNKNOWN"
        )

        issues = []
        shelf_expected = len(expected_products)
        shelf_matched = 0
        revenue_at_risk = 0

        for sku, exp_count in expected_counts.items():
            det_count = detected_counts.get(sku, 0)
            prod_name = next((p["name"] for p in expected_products if p["sku"] == sku), sku)
            prod_price = next((d.get("product_price", 0) for d in detected_on_shelf if d.get("product_sku") == sku), 0)

            if det_count == 0:
                issues.append({"type": "STOCKOUT", "product": prod_name, "expected": exp_count, "found": 0})
                revenue_at_risk += prod_price * exp_count
                all_alerts.append({"type": "STOCKOUT", "shelf": shelf_id, "product": prod_name, "sku": sku,
                                   "expected": exp_count, "found": 0, "revenue": prod_price * exp_count, "priority": "CRITICAL"})
            elif det_count < exp_count:
                missing = exp_count - det_count
                issues.append({"type": "LOW_STOCK", "product": prod_name, "expected": exp_count, "found": det_count})
                revenue_at_risk += prod_price * missing
                all_alerts.append({"type": "LOW_STOCK", "shelf": shelf_id, "product": prod_name, "sku": sku,
                                   "expected": exp_count, "found": det_count, "revenue": prod_price * missing, "priority": "HIGH"})
                shelf_matched += det_count
            else:
                shelf_matched += exp_count

        for sku, count in detected_counts.items():
            if sku not in expected_counts:
                prod_name = next((d.get("product_name", sku) for d in detected_on_shelf if d.get("product_sku") == sku), sku)
                issues.append({"type": "UNAUTHORIZED", "product": prod_name})
                all_alerts.append({"type": "UNAUTHORIZED", "shelf": shelf_id, "product": prod_name, "sku": sku, "priority": "MEDIUM"})

        # Position check
        expected_order = [p["sku"] for p in expected_products if p["sku"] != "UNKNOWN"]
        detected_sorted = sorted(
            [d for d in detected_on_shelf if d.get("product_sku", "UNKNOWN") != "UNKNOWN"],
            key=lambda d: d["bbox"][0]
        )
        detected_order = [d.get("product_sku") for d in detected_sorted]

        if expected_order and detected_order:
            for pos_idx in range(min(len(expected_order), len(detected_order))):
                if expected_order[pos_idx] != detected_order[pos_idx]:
                    exp_name = next((p["name"] for p in expected_products if p["sku"] == expected_order[pos_idx]), "?")
                    det_name = next((d.get("product_name", "?") for d in detected_sorted if d.get("product_sku") == detected_order[pos_idx]), "?")
                    issues.append({"type": "MISPLACED", "product": det_name, "expected_product": exp_name, "position": pos_idx + 1})
                    all_alerts.append({"type": "MISPLACED", "shelf": shelf_id, "product": det_name, "priority": "HIGH"})
                    if shelf_matched > 0:
                        shelf_matched -= 0.5

        if not issues:
            issues.append({"type": "OK", "product": "All products in correct position"})

        comp_pct = (shelf_matched / shelf_expected * 100) if shelf_expected > 0 else 100
        shelf_compliance[shelf_id] = {
            "compliance": round(comp_pct, 1), "expected": shelf_expected,
            "detected": len(detected_on_shelf), "matched": shelf_matched,
            "issues": issues, "revenue_at_risk": revenue_at_risk,
        }
        total_expected += shelf_expected
        total_matched += shelf_matched

    overall = (total_matched / total_expected * 100) if total_expected > 0 else 100
    total_revenue_risk = sum(s["revenue_at_risk"] for s in shelf_compliance.values())

    # Draw annotated frame
    annotated = models.draw_annotated(img, detections)

    # Log to database
    try:
        log_id = log_compliance(
            planogram_name=planogram_name, compliance=round(overall, 1),
            detected=len(detections), expected=total_expected,
            revenue_risk=total_revenue_risk, alert_count=len(all_alerts), scan_number=0,
        )
        for alert in all_alerts:
            log_alert(
                compliance_log_id=log_id, alert_type=alert["type"],
                shelf_id=alert.get("shelf"), product_name=alert["product"],
                product_sku=alert.get("sku", ""), priority=alert["priority"],
                expected_count=alert.get("expected"), found_count=alert.get("found"),
                revenue=alert.get("revenue", 0),
            )
    except Exception:
        pass

    return {
        "overall_compliance": round(overall, 1),
        "total_detected": len(detections),
        "total_expected": total_expected,
        "revenue_at_risk": total_revenue_risk,
        "shelf_compliance": shelf_compliance,
        "alerts": all_alerts,
        "annotated_image": image_to_base64(annotated),
    }


# ── Analytics ─────────────────────────────────────────────

@app.get("/api/analytics/summary")
async def analytics_summary():
    try:
        return get_analytics_summary()
    except Exception:
        return {}


@app.get("/api/compliance/history")
async def compliance_history():
    try:
        return {"logs": get_compliance_logs_as_list()}
    except Exception:
        return {"logs": []}


@app.get("/api/alerts")
async def alerts():
    try:
        return {"alerts": get_alerts_history()}
    except Exception:
        return {"alerts": []}


# ── Push Notifications ────────────────────────────────────

@app.post("/api/notify")
async def send_notification(
    title: str = Form("ShelfMind Alert"),
    message: str = Form(...),
    topic: str = Form("shelfmind-alerts"),
    priority: str = Form("high"),
):
    """Send push notification via ntfy.sh."""
    import requests as req
    try:
        clean_title = title.encode("ascii", "ignore").decode("ascii").strip() or "ShelfMind Alert"
        req.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": clean_title, "Priority": priority, "Tags": "warning"},
            timeout=5,
        )
        return {"status": "sent"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# ── WebSocket Live Monitoring ─────────────────────────────

@app.websocket("/ws/monitor")
async def live_monitor(websocket: WebSocket):
    """Real-time shelf monitoring via WebSocket."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"])
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

                # Resize for speed
                max_dim = 640
                if img.width > max_dim:
                    ratio = max_dim / img.width
                    img = img.resize((max_dim, int(img.height * ratio)), Image.LANCZOS)

                planogram_name = msg.get("planogram", "")
                conf = msg.get("confidence", 0.3)

                # Run compliance if planogram specified
                if planogram_name:
                    planograms = get_planograms()
                    if planogram_name in planograms:
                        # Full compliance (simplified for WebSocket speed)
                        detections = models.detect(img, conf=conf)
                        boundaries, n_shelves = models.detect_shelf_levels(detections, img.height)
                        shelf_assignments = models.assign_to_shelves(detections, boundaries)

                        catalog_products = get_products()
                        for det in detections:
                            x1, y1, x2, y2 = det["bbox"]
                            crop = img.crop((max(0, x1), max(0, y1), x2, y2))
                            emb = models.get_embedding(crop)
                            match, score = models.search_product_faiss(emb, catalog_products, threshold=0.3)
                            if match:
                                det["product_name"] = match["name"]
                                det["product_sku"] = match["sku"]
                                det["match_score"] = round(score, 3)
                                det["status"] = "match"
                            else:
                                det["product_name"] = "Unknown"
                                det["product_sku"] = "UNKNOWN"
                                det["match_score"] = round(score, 3)
                                det["status"] = "unknown"

                        annotated = models.draw_annotated(img, detections)
                        await websocket.send_json({
                            "type": "detection",
                            "total_products": len(detections),
                            "detections": detections,
                            "annotated_image": image_to_base64(annotated),
                        })
                else:
                    # Just detection without compliance
                    detections = models.detect(img, conf=conf)
                    annotated = models.draw_annotated(img, [
                        {**d, "product_name": f"P{i+1}", "match_score": d["confidence"], "status": "match"}
                        for i, d in enumerate(detections)
                    ])
                    await websocket.send_json({
                        "type": "detection",
                        "total_products": len(detections),
                        "annotated_image": image_to_base64(annotated),
                    })

            elif msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting ShelfMind AI on http://0.0.0.0:{port}")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
