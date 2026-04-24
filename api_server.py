"""
=============================================================================
ShelfMind AI — FastAPI Backend Server
=============================================================================
Production-ready API serving YOLO26s detection, DINOv2 embeddings,
FAISS product matching, and LightGBM demand forecasting.

Run: python api_server.py
Open: http://localhost:7860
=============================================================================
"""

import os
import sys
import io
import gc
import json
import time
import base64
import hashlib
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F

# ── FastAPI ───────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "shelfmind_models_yolo26"
LIGHTNING_DIR = ROOT / "shelfmind_models_lightning"
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
DB_DIR = DATA_DIR
CATALOG_DIR = DATA_DIR / "store_catalog"
REF_IMG_DIR = CATALOG_DIR / "reference_images"

for d in [STATIC_DIR, DATA_DIR, CATALOG_DIR, REF_IMG_DIR]:
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
        self.faiss_index = None
        self.faiss_metadata = None
        self.forecast_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False

    def load_all(self):
        """Load all models at startup."""
        start = time.time()
        logger.info(f"Loading models on device={self.device}...")

        # 1. YOLO26s Detection
        self._load_yolo()

        # 2. DINOv2 Embeddings
        self._load_dinov2()

        # 3. FAISS Index
        self._load_faiss()

        # 4. LightGBM Forecast
        self._load_forecast()

        self.ready = True
        elapsed = time.time() - start
        logger.info(f"[OK] All models loaded in {elapsed:.1f}s")

    def _load_yolo(self):
        yolo_path = MODEL_DIR / "yolo_shelf_best.pt"
        if not yolo_path.exists():
            # Fallback: check root
            yolo_path = ROOT / "yolo26s.pt"
        if yolo_path.exists():
            from ultralytics import YOLO
            self.yolo = YOLO(str(yolo_path))
            logger.info(f"  [OK] YOLO26s: {yolo_path.name} ({yolo_path.stat().st_size/1e6:.1f} MB)")
        else:
            logger.warning("  [SKIP] YOLO model not found")

    def _load_dinov2(self):
        dinov2_path = LIGHTNING_DIR / "shelfmind_output_dinov2_shelf_finetuned.pth"
        if not dinov2_path.exists():
            dinov2_path = MODEL_DIR / "dinov2_shelf_finetuned.pth"

        try:
            self.dinov2 = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=True, verbose=False
            )
            if dinov2_path.exists():
                state = torch.load(str(dinov2_path), map_location=self.device, weights_only=True)
                self.dinov2.load_state_dict(state, strict=False)
                logger.info(f"  [OK] DINOv2: fine-tuned ({dinov2_path.stat().st_size/1e6:.1f} MB)")
            else:
                logger.info("  [OK] DINOv2: pretrained (no fine-tuned weights found)")
            self.dinov2.eval().to(self.device)
        except Exception as e:
            logger.warning(f"  [SKIP] DINOv2 load error: {e}")
            self.dinov2 = None

    def _load_faiss(self):
        faiss_path = MODEL_DIR / "sku_faiss_index.bin"
        meta_path = MODEL_DIR / "sku_metadata.json"
        if faiss_path.exists():
            try:
                import faiss
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"  [OK] FAISS: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"  [SKIP] FAISS error: {e}")
        if meta_path.exists():
            with open(meta_path) as f:
                self.faiss_metadata = json.load(f)
            logger.info(f"  [OK] FAISS metadata: {len(self.faiss_metadata)} entries")

    def _load_forecast(self):
        forecast_path = MODEL_DIR / "lgbm_forecast_model.pkl"
        if forecast_path.exists():
            try:
                import joblib
                self.forecast_model = joblib.load(str(forecast_path))
                logger.info(f"  [OK] LightGBM forecast model loaded")
            except Exception as e:
                logger.warning(f"  [SKIP] Forecast model error: {e}")

    # ── Inference Methods ─────────────────────────────────

    def detect(self, image: Image.Image, conf: float = 0.3) -> dict:
        """Run YOLO26s detection on an image."""
        if not self.yolo:
            raise RuntimeError("YOLO model not loaded")

        start = time.time()
        results = self.yolo.predict(
            np.array(image), conf=conf, verbose=False, imgsz=640
        )
        elapsed_ms = (time.time() - start) * 1000

        boxes = results[0].boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(confs[i]), 3),
                    "class": "product",
                })

        return {
            "total_products": len(detections),
            "detections": detections,
            "inference_ms": round(elapsed_ms, 1),
        }

    def get_embedding(self, crop: Image.Image) -> np.ndarray:
        """Get DINOv2 embedding for a product crop."""
        if not self.dinov2:
            return np.zeros(768, dtype=np.float32)

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
            emb = self.dinov2(tensor)
        return emb.cpu().numpy().flatten()

    def match_product(self, embedding: np.ndarray, top_k: int = 5) -> list:
        """Match embedding against FAISS index."""
        if self.faiss_index is None:
            return []

        embedding = embedding.reshape(1, -1).astype(np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        distances, indices = self.faiss_index.search(embedding, top_k)

        matches = []
        for i in range(min(top_k, len(indices[0]))):
            idx = int(indices[0][i])
            if idx < 0 or (self.faiss_metadata and idx >= len(self.faiss_metadata)):
                continue
            score = float(distances[0][i])
            meta = self.faiss_metadata[idx] if self.faiss_metadata else {"crop_id": f"unknown_{idx}"}
            matches.append({
                "rank": i + 1,
                "score": round(score, 4),
                "metadata": meta,
            })
        return matches

    def analyze_shelf(self, image: Image.Image, conf: float = 0.3) -> dict:
        """Full pipeline: detect → crop → embed → match."""
        start = time.time()

        # Step 1: Detect
        detection_result = self.detect(image, conf)
        detections = detection_result["detections"]

        # Step 2: Crop + Embed + Match each product
        products = []
        for det in detections[:50]:  # Cap at 50 for speed
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((x1, y1, x2, y2))

            # Get embedding
            emb = self.get_embedding(crop)

            # Match
            matches = self.match_product(emb, top_k=3)

            products.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "matches": matches,
            })

        elapsed_ms = (time.time() - start) * 1000

        return {
            "total_products": detection_result["total_products"],
            "identified_products": len([p for p in products if p["matches"]]),
            "products": products,
            "inference_ms": round(elapsed_ms, 1),
        }

    def draw_detections(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes on image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)

        colors = ["#00d4aa", "#00b4d8", "#7b68ee", "#ff6b6b", "#ffe66d"]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label = f'{det["confidence"]:.0%}'
            draw.rectangle([x1, y1 - 16, x1 + 45, y1], fill=color)
            draw.text((x1 + 2, y1 - 14), label, fill="black")

        return img


# =============================================================================
# DATABASE LAYER (SQLite — reuse existing db.py)
# =============================================================================
sys.path.insert(0, str(ROOT / "app"))
try:
    from db import (
        setup_database, add_product, get_products, get_product_count,
        get_next_product_id, delete_product, clear_all_products,
        log_compliance, log_alert, get_compliance_logs_as_list,
        get_analytics_summary, get_alerts_history,
    )
    setup_database()
    logger.info("[OK] Database initialized")
except ImportError:
    logger.warning("[SKIP] db.py not found, using in-memory store")
    # Minimal fallback
    _products = []
    def get_products(): return _products
    def add_product(**kwargs): _products.append(kwargs)
    def get_product_count(): return len(_products)
    def delete_product(pid): pass

# =============================================================================
# FASTAPI APP
# =============================================================================
models = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    models.load_all()
    yield
    # Cleanup
    del models.yolo, models.dinov2
    torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(
    title="ShelfMind AI",
    description="Smart Retail Shelf Intelligence API",
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

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Helper ────────────────────────────────────────────────
def image_to_base64(img: Image.Image, fmt="JPEG", quality=85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def read_upload_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>ShelfMind AI — API Running</h1><p>Place index.html in /static/</p>")


@app.get("/api/health")
async def health():
    """System health check."""
    return {
        "status": "healthy" if models.ready else "loading",
        "device": models.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models": {
            "yolo": models.yolo is not None,
            "dinov2": models.dinov2 is not None,
            "faiss": models.faiss_index is not None,
            "faiss_vectors": models.faiss_index.ntotal if models.faiss_index else 0,
            "forecast": models.forecast_model is not None,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/detect")
async def detect_products(
    image: UploadFile = File(...),
    confidence: float = Form(0.3),
    annotate: bool = Form(True),
):
    """Detect products in a shelf image using YOLO26s."""
    contents = await image.read()
    img = read_upload_image(contents)

    result = models.detect(img, conf=confidence)

    response = {
        "total_products": result["total_products"],
        "detections": result["detections"],
        "inference_ms": result["inference_ms"],
        "image_size": {"width": img.width, "height": img.height},
    }

    if annotate and result["detections"]:
        annotated = models.draw_detections(img, result["detections"])
        response["annotated_image"] = image_to_base64(annotated)

    return response


@app.post("/api/analyze")
async def analyze_shelf(
    image: UploadFile = File(...),
    confidence: float = Form(0.3),
):
    """Full shelf analysis: detect → embed → match → compliance."""
    contents = await image.read()
    img = read_upload_image(contents)

    result = models.analyze_shelf(img, conf=confidence)

    # Draw annotated image
    annotated = models.draw_detections(img, [
        {"bbox": p["bbox"], "confidence": p["confidence"]}
        for p in result["products"]
    ])

    return {
        "total_products": result["total_products"],
        "identified_products": result["identified_products"],
        "compliance_score": round(
            result["identified_products"] / max(result["total_products"], 1) * 100, 1
        ),
        "products": result["products"][:20],  # Limit response size
        "inference_ms": result["inference_ms"],
        "annotated_image": image_to_base64(annotated),
        "image_size": {"width": img.width, "height": img.height},
    }


@app.post("/api/match")
async def match_product(image: UploadFile = File(...)):
    """Match a product crop against the FAISS index."""
    contents = await image.read()
    crop = read_upload_image(contents)

    emb = models.get_embedding(crop)
    matches = models.match_product(emb, top_k=5)

    return {
        "matches": matches,
        "embedding_dim": len(emb),
    }


# ── Product Catalog CRUD ──────────────────────────────────

@app.get("/api/products")
async def list_products():
    """List all products in catalog."""
    products = get_products()
    return {"products": [dict(p) if hasattr(p, 'keys') else p for p in products]}


@app.post("/api/products")
async def create_product(
    name: str = Form(...),
    sku: str = Form(""),
    category: str = Form("General"),
    price: float = Form(0.0),
    image: Optional[UploadFile] = File(None),
):
    """Add a product to the catalog."""
    sku = sku or f"SKU-{hashlib.md5(name.encode()).hexdigest()[:8].upper()}"

    image_path = None
    if image:
        contents = await image.read()
        img = read_upload_image(contents)
        img_filename = f"{sku}.jpg"
        img_path = REF_IMG_DIR / img_filename
        img.save(str(img_path), "JPEG", quality=90)
        image_path = str(img_path)

    try:
        add_product(
            sku=sku, name=name, category=category,
            price=price, image_path=image_path
        )
        return {"status": "ok", "sku": sku, "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/products/{product_id}")
async def remove_product(product_id: int):
    """Delete a product."""
    delete_product(product_id)
    return {"status": "ok"}


@app.post("/api/products/voice")
async def voice_product(transcript: str = Form(...)):
    """Parse voice transcript into product fields."""
    text = transcript.lower().strip()

    # Simple NLP parsing
    name = ""
    category = "General"
    price = 0.0

    # Extract price (look for numbers near "rupees", "rs", "₹", or just numbers)
    import re
    price_match = re.search(r'(?:price|rs|rupees?|₹)\s*(\d+(?:\.\d+)?)', text)
    if not price_match:
        price_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:rupees?|rs)', text)
    if price_match:
        price = float(price_match.group(1))

    # Extract category
    categories = {
        "snack": "Snacks", "biscuit": "Snacks", "chip": "Snacks", "namkeen": "Snacks",
        "drink": "Beverages", "juice": "Beverages", "water": "Beverages", "cola": "Beverages", "soda": "Beverages",
        "dairy": "Dairy", "milk": "Dairy", "curd": "Dairy", "paneer": "Dairy", "cheese": "Dairy",
        "soap": "Personal Care", "shampoo": "Personal Care", "cream": "Personal Care", "toothpaste": "Personal Care",
        "detergent": "Household", "cleaner": "Household", "wash": "Household",
        "rice": "Staples", "flour": "Staples", "atta": "Staples", "dal": "Staples", "oil": "Staples", "sugar": "Staples",
        "chocolate": "Confectionery", "candy": "Confectionery", "toffee": "Confectionery",
        "spice": "Spices", "masala": "Spices",
    }

    for keyword, cat in categories.items():
        if keyword in text:
            category = cat
            break

    # Category from explicit mention
    cat_match = re.search(r'category\s+(\w+)', text)
    if cat_match:
        raw_cat = cat_match.group(1).capitalize()
        for keyword, cat in categories.items():
            if keyword in raw_cat.lower():
                category = cat
                break

    # Name = everything minus price/category parts
    name_text = text
    for pattern in [
        r'(?:price|rs|rupees?|₹)\s*\d+(?:\.\d+)?',
        r'\d+(?:\.\d+)?\s*(?:rupees?|rs)',
        r'category\s+\w+',
    ]:
        name_text = re.sub(pattern, '', name_text, flags=re.I)

    name = name_text.strip().strip(',').strip().title()
    if not name:
        name = "Unknown Product"

    return {
        "parsed": {
            "name": name,
            "category": category,
            "price": price,
            "sku": f"SKU-{hashlib.md5(name.encode()).hexdigest()[:8].upper()}",
        },
        "original_transcript": transcript,
    }


# ── Compliance ────────────────────────────────────────────

@app.get("/api/compliance/history")
async def compliance_history():
    """Get compliance check history."""
    try:
        logs = get_compliance_logs_as_list()
        return {"logs": logs}
    except Exception:
        return {"logs": []}


# ── Forecast ──────────────────────────────────────────────

@app.post("/api/forecast")
async def demand_forecast(
    product_name: str = Form("Generic Product"),
    days: int = Form(7),
):
    """LightGBM demand forecast."""
    if not models.forecast_model:
        raise HTTPException(status_code=503, detail="Forecast model not loaded")

    try:
        forecasts = []
        for d in range(days):
            future_date = datetime.now() + timedelta(days=d)
            features = np.array([[
                future_date.weekday(),
                future_date.month,
                future_date.day,
                1 if future_date.weekday() >= 5 else 0,
                np.random.uniform(50, 200),  # Simulated price
                0,  # SNAP
                np.random.uniform(10, 50),  # Lag features
                np.random.uniform(10, 50),
                np.random.uniform(10, 50),
            ]])
            pred = float(models.forecast_model.predict(features)[0])
            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "day": future_date.strftime("%A"),
                "predicted_demand": round(max(0, pred), 1),
            })

        return {
            "product": product_name,
            "forecast_days": days,
            "predictions": forecasts,
            "total_demand": round(sum(f["predicted_demand"] for f in forecasts), 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket Live Monitoring ─────────────────────────────

@app.websocket("/ws/live")
async def live_monitor(websocket: WebSocket):
    """Real-time shelf monitoring via WebSocket."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive frame as base64
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                # Decode image
                img_data = base64.b64decode(msg["data"])
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

                # Run detection
                result = models.detect(img, conf=0.3)

                # Draw annotations
                annotated = models.draw_detections(img, result["detections"])

                # Send back
                await websocket.send_json({
                    "type": "detection",
                    "total_products": result["total_products"],
                    "inference_ms": result["inference_ms"],
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
    logger.info(f"Starting ShelfMind AI on http://localhost:{port}")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
