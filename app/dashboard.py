"""
ShelfMind AI — Smart Retail Shelf Intelligence
================================================
Complete retail shelf monitoring and inventory optimization system.
Run: streamlit run app/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys
import io
import time
import base64
import hashlib
import re
import requests
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Database module
from db import (
    setup_database, add_product, get_products, get_product_count,
    get_next_product_id, delete_product, clear_all_products, get_catalog_as_dict,
    save_planogram_db, get_planograms, delete_planogram,
    log_compliance, log_alert, get_compliance_logs_as_list,
    get_analytics_summary, get_alerts_history,
)

# ── Path Setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models" / "shelfmind_models"
VIZ_DIR = ROOT / "models" / "training_visualizations"
CATALOG_DIR = ROOT / "data" / "store_catalog"
REF_IMG_DIR = CATALOG_DIR / "reference_images"
PLANOGRAM_DIR = ROOT / "data" / "store_planograms"
COMPLIANCE_DIR = ROOT / "data" / "compliance_logs"
FORECAST_MODEL_PATH = MODEL_DIR / "lgbm_forecast_model.pkl"  # Kept for reference, unused

# Ensure directories exist
for d in [CATALOG_DIR, REF_IMG_DIR, PLANOGRAM_DIR, COMPLIANCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize SQLite database
setup_database()

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShelfMind AI — Smart Retail Shelf Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS Theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #1b1b3a 100%);
        font-family: 'Inter', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding: 1rem 2rem; max-width: 1400px; }

    /* ─── Tab Styling (Pill Navigation) ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 6px 8px;
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 12px 24px;
        color: #8892b0;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ccd6f6;
        background: rgba(255,255,255,0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%) !important;
        color: #0a0a1a !important;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(0,212,170,0.3);
        border: 1px solid rgba(0,212,170,0.3);
    }

    /* ─── Glassmorphism Metric Cards ─── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px 16px;
        text-align: center;
        backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #00b4d8, #7b68ee);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0,212,170,0.25);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,212,170,0.12);
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6b7b9e;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* ─── Product Cards ─── */
    .product-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .product-card:hover {
        border-color: rgba(0,180,216,0.3);
        box-shadow: 0 8px 24px rgba(0,180,216,0.08);
        transform: translateY(-2px);
    }
    .product-card img { border-radius: 12px; }

    /* ─── Alert Cards ─── */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255,67,67,0.12), rgba(255,67,67,0.03));
        border-left: 4px solid #ff4343;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #ff8a8a;
        font-size: 14px;
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease-out;
    }
    .alert-warning {
        background: linear-gradient(135deg, rgba(255,170,0,0.12), rgba(255,170,0,0.03));
        border-left: 4px solid #ffaa00;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #ffd066;
        font-size: 14px;
        backdrop-filter: blur(10px);
    }
    .alert-ok {
        background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(0,212,170,0.03));
        border-left: 4px solid #00d4aa;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #66ffd9;
        font-size: 14px;
        backdrop-filter: blur(10px);
    }
    .alert-info {
        background: linear-gradient(135deg, rgba(0,180,216,0.12), rgba(0,180,216,0.03));
        border-left: 4px solid #00b4d8;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #66d9f0;
        font-size: 14px;
        backdrop-filter: blur(10px);
    }

    /* ─── Section Headers ─── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e6f1ff, #ccd6f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 20px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.3px;
    }

    /* ─── Status Badges ─── */
    .badge-ok { background: #00d4aa18; color: #00d4aa; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #00d4aa30; }
    .badge-warn { background: #ffaa0018; color: #ffaa00; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #ffaa0030; }
    .badge-critical { background: #ff434318; color: #ff4343; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #ff434330; }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b1b3a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ─── Hero Section ─── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7b68ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #6b7b9e;
        font-size: 0.95rem;
        margin-bottom: 24px;
        line-height: 1.6;
    }

    /* ─── Buttons ─── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0,212,170,0.25);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(0,212,170,0.35) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: #ccd6f6 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(0,212,170,0.3) !important;
    }

    /* ─── Inputs ─── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e6f1ff !important;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00d4aa !important;
        box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
    }

    /* ─── Radio Buttons ─── */
    .stRadio > div {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        font-weight: 600;
    }

    /* ─── Images ─── */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ─── Dividers ─── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 20px 0 !important;
    }

    /* ─── Footer ─── */
    .footer {
        text-align: center;
        padding: 32px;
        color: #4a5568;
        font-size: 12px;
        margin-top: 48px;
        border-top: 1px solid rgba(255,255,255,0.04);
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ─── Animations ─── */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ─── Better DataFrames ─── */
    .stDataFrame { border-radius: 16px; overflow: hidden; }

    /* ─── Form Container ─── */
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 ShelfMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Smart Retail Shelf Intelligence — Computer Vision-Driven Inventory Monitoring & Demand Optimization</div>', unsafe_allow_html=True)

# ── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    """Load YOLO model for product detection."""
    try:
        from ultralytics import YOLO
        # v2 fine-tuned model (1280px, 60ep, mAP50=91.7%)
        model_path_v2 = ROOT / "models" / "runs_detect_shelfmind_models_yolo26s_1280_v2_weights_best.pt"
        # v1 fallback
        model_path_v1 = MODEL_DIR / "yolo_shelf_best.pt"
        if model_path_v2.exists():
            model = YOLO(str(model_path_v2))
            model.to("cpu")
            print(f"[OK] YOLO26s v2 loaded (1280px, mAP50=91.7%)")
            return model
        elif model_path_v1.exists():
            model = YOLO(str(model_path_v1))
            model.to("cpu")
            return model
        # Fallback to pretrained YOLO26s
        model = YOLO("yolo26s.pt")
        model.to("cpu")
        return model
    except Exception as e:
        st.error(f"YOLO load failed: {e}")
        return None

@st.cache_resource
def load_rfdetr():
    """Load fine-tuned RF-DETR model for product detection (PyTorch Lightning checkpoint)."""
    rfdetr_path = ROOT / "shelfmind_models_lightning" / "shelfmind_output_rfdetr_shelf_best.pt"
    if not rfdetr_path.exists():
        print(f"[WARNING] RF-DETR model not found at {rfdetr_path}")
        return None
    try:
        from rfdetr import RFDETRBase
        model = RFDETRBase(pretrain_weights=str(rfdetr_path))
        print(f"[OK] RF-DETR loaded ({rfdetr_path.stat().st_size/1e6:.1f} MB)")
        return model
    except TypeError:
        try:
            from rfdetr import RFDETRBase
            model = RFDETRBase(checkpoint=str(rfdetr_path))
            print(f"[OK] RF-DETR loaded ({rfdetr_path.stat().st_size/1e6:.1f} MB)")
            return model
        except Exception as e:
            print(f"[ERROR] RF-DETR load failed: {e}")
            return None
    except ImportError:
        print("[ERROR] rfdetr package not installed")
        return None
    except Exception as e:
        print(f"[ERROR] RF-DETR load failed: {e}")
        return None

@st.cache_resource
def load_dinov2():
    """Load pretrained DINOv2 for product embedding."""
    try:
        import torch
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        model = model.to("cpu")
        model.eval()
        return model
    except Exception as e:
        st.error(f"DINOv2 load failed: {e}")
        return None

@st.cache_resource
def load_dinov2_finetuned():
    """Load fine-tuned DINOv2 with projector for product embedding.
    Returns (backbone, projector) tuple.
    """
    import torch
    import torch.nn as nn
    finetuned_path = ROOT / "_output_" / "dinov2_shelf_finetuned.pth"
    projector_path = ROOT / "_output_" / "dinov2_projector.pth"
    try:
        # Load base model
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        if finetuned_path.exists():
            state = torch.load(str(finetuned_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[OK] DINOv2 fine-tuned backbone loaded ({finetuned_path.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"[WARNING] Fine-tuned weights not found at {finetuned_path}")
            return None
        model = model.to("cpu")
        model.eval()

        # Build and load projector: 768 → 2048 → 2048 → 256
        projector = None
        if projector_path.exists():
            projector = nn.Sequential(
                nn.Linear(768, 2048),       # net.0
                nn.BatchNorm1d(2048),       # net.1
                nn.ReLU(inplace=True),      # net.2
                nn.Linear(2048, 2048),      # net.3
                nn.BatchNorm1d(2048),       # net.4
                nn.ReLU(inplace=True),      # net.5
                nn.Linear(2048, 256),       # net.6
            )
            proj_state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
            # Strip 'net.' prefix: checkpoint has 'net.0.weight' but Sequential expects '0.weight'
            cleaned_state = {k.replace("net.", "", 1): v for k, v in proj_state.items()}
            projector.load_state_dict(cleaned_state, strict=True)
            projector = projector.to("cpu")
            projector.eval()
            print(f"[OK] DINOv2 projector loaded (768→256, {projector_path.stat().st_size/1e6:.1f} MB)")

        return (model, projector)
    except Exception as e:
        print(f"[ERROR] DINOv2 fine-tuned load failed: {e}")
        return None

def get_embedding(model_or_tuple, image):
    """Get DINOv2 embedding for an image.
    Accepts either a plain model or a (backbone, projector) tuple from load_dinov2_finetuned.
    """
    import torch
    from torchvision import transforms

    # Unpack if tuple (fine-tuned model with projector)
    if isinstance(model_or_tuple, tuple):
        model, projector = model_or_tuple
    else:
        model, projector = model_or_tuple, None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to("cpu")
    with torch.no_grad():
        backbone_emb = model(img_tensor)  # (1, 768) or (1, 384)
        if projector is not None:
            embedding = projector(backbone_emb).squeeze().numpy()  # (256,)
        else:
            embedding = backbone_emb.squeeze().numpy()
    return embedding / np.linalg.norm(embedding)  # L2 normalize


def get_robust_embedding(model, image, return_views=False):
    """Generate robust embedding by averaging 15 augmented views of 1 photo.
    
    Invariance types covered:
      - Translational: center, left, right, top, bottom crops
      - Scale: zoom in (80%) and zoom out (110%)
      - Rotational: ±5° slight rotation
      - Photometric: brightness ±20%, contrast +20%, saturation ±20%
      - Mirror: horizontal flip
      - Blur: Gaussian blur (simulates out-of-focus / motion blur)
    
    If return_views=True, returns (embedding, views_list, view_names)
    """
    import torch
    from torchvision import transforms
    from PIL import ImageEnhance, ImageOps, ImageFilter
    
    img = image.convert("RGB")
    w, h = img.size
    
    view_names = []
    views = []
    
    # 1. Original
    views.append(img)
    view_names.append("Original")
    
    # 2. Horizontal flip (mirror invariance)
    views.append(ImageOps.mirror(img))
    view_names.append("H-Flip")
    
    # 3. Center crop 80% (scale invariance - zoom in)
    mw, mh = int(w * 0.1), int(h * 0.1)
    views.append(img.crop((mw, mh, w - mw, h - mh)))
    view_names.append("Center 80%")
    
    # 4. Left-shifted crop (translational invariance)
    views.append(img.crop((0, mh, int(w * 0.85), h - mh)))
    view_names.append("Left Crop")
    
    # 5. Right-shifted crop (translational invariance)
    views.append(img.crop((int(w * 0.15), mh, w, h - mh)))
    view_names.append("Right Crop")
    
    # 6. Top crop (translational invariance)
    views.append(img.crop((mw, 0, w - mw, int(h * 0.85))))
    view_names.append("Top Crop")
    
    # 7. Bottom crop (translational invariance — was missing)
    views.append(img.crop((mw, int(h * 0.15), w - mw, h)))
    view_names.append("Bottom Crop")
    
    # 8. Slight rotation +5° (rotational invariance)
    views.append(img.rotate(-5, expand=False, fillcolor=(128, 128, 128)))
    view_names.append("Rotate +5°")
    
    # 9. Slight rotation -5° (rotational invariance)
    views.append(img.rotate(5, expand=False, fillcolor=(128, 128, 128)))
    view_names.append("Rotate -5°")
    
    # 10. Brightness +20% (photometric invariance)
    views.append(ImageEnhance.Brightness(img).enhance(1.2))
    view_names.append("Bright +20%")
    
    # 11. Brightness -20% (shadow/dark shelf conditions)
    views.append(ImageEnhance.Brightness(img).enhance(0.8))
    view_names.append("Bright -20%")
    
    # 12. Contrast +20% (photometric invariance)
    views.append(ImageEnhance.Contrast(img).enhance(1.2))
    view_names.append("Contrast +20%")
    
    # 13. Saturation +20% (color temperature variation)
    views.append(ImageEnhance.Color(img).enhance(1.2))
    view_names.append("Saturation +20%")
    
    # 14. Saturation -20% (faded/washed-out lighting)
    views.append(ImageEnhance.Color(img).enhance(0.8))
    view_names.append("Saturation -20%")
    
    # 15. Gaussian blur (out-of-focus / motion blur during live scan)
    views.append(img.filter(ImageFilter.GaussianBlur(radius=1.5)))
    view_names.append("Blur σ=1.5")
    
    # Compute embeddings for all views and average
    embeddings = []
    for view in views:
        emb = get_embedding(model, view)
        embeddings.append(emb)
    
    avg_emb = np.mean(embeddings, axis=0)
    normalized = avg_emb / np.linalg.norm(avg_emb)  # L2 normalize
    
    if return_views:
        return normalized, views, view_names
    return normalized


# ── Auto-Crop & OCR Helpers ───────────────────────────────────────────────
def auto_crop_product(image, yolo_model, conf=0.3, padding=5):
    """Auto-crop product from image using YOLO → GrabCut fallback.
    
    Strategy:
      1. Try YOLO detection (works for shelf/multi-product scenes)
      2. If YOLO misses → use GrabCut foreground segmentation
         (works for single product close-ups on table/plain background)
    
    Returns: (cropped_image, bbox) or (original, None)
    """
    import cv2
    img_np = np.array(image)
    
    # ── Step 1: Try YOLO detection ────────────────────────────────────
    results = yolo_model(img_np, conf=conf, imgsz=640, verbose=False)

    best_crop = None
    best_area = 0
    best_bbox = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w = img_np.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_crop = image.crop((x1, y1, x2, y2))
                best_bbox = (x1, y1, x2, y2)

    if best_crop:
        # ── Refine YOLO crop with GrabCut for tighter boundary ──
        try:
            crop_np = cv2.cvtColor(np.array(best_crop), cv2.COLOR_RGB2BGR)
            ch, cw = crop_np.shape[:2]
            if ch > 50 and cw > 50:  # Only refine if crop is large enough
                margin_x = max(5, int(cw * 0.05))
                margin_y = max(5, int(ch * 0.05))
                rect = (margin_x, margin_y, cw - 2 * margin_x, ch - 2 * margin_y)
                mask = np.zeros((ch, cw), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                cv2.grabCut(crop_np, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, rw, rh = cv2.boundingRect(largest)
                    refine_ratio = (rw * rh) / (cw * ch)
                    if 0.20 < refine_ratio < 0.95:  # Reasonable refinement
                        rx1 = max(0, x - 3)
                        ry1 = max(0, y - 3)
                        rx2 = min(cw, x + rw + 3)
                        ry2 = min(ch, y + rh + 3)
                        refined = best_crop.crop((rx1, ry1, rx2, ry2))
                        return refined, best_bbox
        except Exception:
            pass  # Fall back to YOLO crop if GrabCut fails
        return best_crop, best_bbox

    # ── Step 2: GrabCut foreground segmentation (fast) ─────────────────
    # Downsize for speed, then map bbox back to original
    try:
        h, w = img_np.shape[:2]
        max_dim = 480
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                               (int(w * scale), int(h * scale)))
        else:
            small = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        sh, sw = small.shape[:2]
        margin_x, margin_y = int(sw * 0.10), int(sh * 0.10)
        rect = (margin_x, margin_y, sw - 2 * margin_x, sh - 2 * margin_y)

        mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            crop_ratio = (cw * ch) / (sw * sh)

            # Skip crop if product already fills >60% of frame
            # (cropping would cut off parts of the product)
            if crop_ratio > 0.60:
                return image, None

            if crop_ratio > 0.05:
                inv = 1.0 / scale
                pad = padding
                x1 = max(0, int(x * inv) - pad)
                y1 = max(0, int(y * inv) - pad)
                x2 = min(w, int((x + cw) * inv) + pad)
                y2 = min(h, int((y + ch) * inv) + pad)
                cropped = image.crop((x1, y1, x2, y2))
                return cropped, (x1, y1, x2, y2)
    except Exception:
        pass
    
    return image, None


def detect_all_products(image, det_model, conf=0.3, padding=3):
    """Detect ALL products in a shelf image. Supports both YOLO (ultralytics) and RF-DETR (rfdetr package).
    Returns list of (crop, bbox, confidence).
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    crops = []

    # Check if this is an rfdetr model
    is_rfdetr = 'RFDETR' in type(det_model).__name__

    if is_rfdetr:
        # RF-DETR predict returns a supervision.Detections object
        try:
            detections = det_model.predict(image, threshold=conf)
            # supervision.Detections has .xyxy (ndarray) and .confidence (ndarray)
            boxes = detections.xyxy        # shape (N, 4) — [x1,y1,x2,y2]
            scores = detections.confidence  # shape (N,)
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    crop = image.crop((x1, y1, x2, y2))
                    bbox = (x1, y1, x2, y2)
                    score = float(scores[i]) if scores is not None else 1.0
                    crops.append((crop, bbox, score))
        except Exception as e:
            st.error(f"RF-DETR prediction error: {e}")
    else:
        # Ultralytics YOLO API
        results = det_model(img_np, conf=conf, imgsz=1280, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                crop = image.crop((x1, y1, x2, y2))
                bbox = (x1, y1, x2, y2)
                crops.append((crop, bbox, float(box.conf[0])))

    return crops


@st.cache_resource
def load_ocr():
    """Load EasyOCR reader for product text recognition."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return reader
    except Exception as e:
        st.warning(f"OCR not available: {e}")
        return None


def extract_text_from_crop(ocr_reader, crop_image):
    """Extract product name + quantity from crop using OCR.
    
    Strategy:
      1. Find the LARGEST text on the label (= brand name, in big font)
      2. Extract quantity pattern (2.25L, 330ml, 500g, etc.)
      3. Return: "BrandName Quantity"
    """
    if ocr_reader is None:
        return ""
    try:
        # Resize for speed — OCR is very slow on full-res phone images
        img = crop_image.copy()
        max_dim = 640
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        img_np = np.array(img)
        results = ocr_reader.readtext(img_np, detail=1, paragraph=False)
        if not results:
            return ""

        # ── Step 1: Find brand name (largest text by bbox area) ────────
        brand_name = ""
        max_area = 0
        for (bbox, text, conf) in results:
            text = text.strip()
            if conf < 0.2 or len(text) < 2:
                continue
            # Skip common non-brand words
            skip_words = {"ingredients", "nutrition", "serving", "energy",
                          "protein", "sugar", "sodium", "carbohydrate", "fat",
                          "total", "added", "manufactured", "contains", "per",
                          "information", "flavours", "natural", "artificial",
                          "carbonated", "water", "values", "approximate"}
            if text.lower() in skip_words:
                continue
            # Calculate bbox area (larger text = more prominent = brand name)
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > max_area:
                max_area = area
                brand_name = text

        # ── Step 2: Extract quantity pattern (2.25L, 330ml, 500g) ─────
        quantity = ""
        all_text = " ".join([t for (_, t, c) in results if c > 0.2])
        qty_match = re.search(
            r'(\d+\.?\d*)\s*(ml|ML|mL|ltr|LTR|Ltr|[lL]|[gG]|[kK][gG])\b',
            all_text
        )
        if qty_match:
            quantity = qty_match.group(0).strip()

        # ── Combine ───────────────────────────────────────────────────
        parts = [p for p in [brand_name, quantity] if p]
        return " ".join(parts) if parts else ""
    except Exception:
        return ""


def scan_barcode(image):
    """Detect and decode barcodes using OpenCV's built-in BarcodeDetector.
    Tries multiple rotations and preprocessing for robustness.
    Returns barcode string or None.
    """
    try:
        import cv2
        detector = cv2.barcode.BarcodeDetector()
        img_np = np.array(image)

        # Try 4 rotations × 2 preprocessing methods
        for rotation in [0, 90, 180, 270]:
            if rotation == 0:
                rotated = img_np
            else:
                rotated = np.array(image.rotate(-rotation, expand=True))

            # Method 1: Original
            gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
            ok, decoded_info, _, _ = detector.detectAndDecodeMulti(gray)
            if ok and decoded_info is not None:
                for info in decoded_info:
                    if info and len(info) >= 4:
                        return info

            # Method 2: Enhanced contrast
            gray_eq = cv2.equalizeHist(gray)
            ok, decoded_info, _, _ = detector.detectAndDecodeMulti(gray_eq)
            if ok and decoded_info is not None:
                for info in decoded_info:
                    if info and len(info) >= 4:
                        return info

        return None
    except Exception:
        return None


def lookup_barcode_info(barcode_number):
    """Lookup product info from barcode using Open Food Facts API (free).
    Note: Many Indian products are NOT in the database — that's normal.
    Returns dict with name, category, brand or empty dict.
    """
    try:
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_number}.json"
        headers = {"User-Agent": "ShelfMind/1.0 (retail-compliance)"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200 and resp.text:
            import json
            try:
                data = json.loads(resp.text)
            except json.JSONDecodeError:
                return {}
            if data.get("status") == 1:
                product = data.get("product", {})
                name = product.get("product_name", "")
                brand = product.get("brands", "")
                category = product.get("categories", "").split(",")[0].strip() if product.get("categories") else ""
                quantity = product.get("quantity", "")
                # Build a useful product name
                full_name = ""
                if brand and name:
                    full_name = f"{brand} {name}"
                elif name:
                    full_name = name
                elif brand:
                    full_name = brand
                if quantity and quantity not in full_name:
                    full_name = f"{full_name} {quantity}".strip()
                return {
                    "name": full_name,
                    "category": category,
                    "brand": brand,
                    "quantity": quantity,
                }
    except Exception:
        pass
    return {}


def get_crop_geometry(bbox):
    """Get geometry features from bounding box for size-based discrimination."""
    if bbox is None:
        return {"width": 0, "height": 0, "aspect_ratio": 0, "area": 0}
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(h / max(w, 1), 2),
        "area": w * h,
    }


def cluster_unique_products(crops_data, dinov2_model, similarity_threshold=0.85):
    """Cluster detected crops using HDBSCAN with size-aware distance fusion.
    
    HDBSCAN: Automatically determines the optimal number of clusters without
    requiring a similarity threshold. Finds dense regions in embedding space
    and treats sparse points as noise (assigned to nearest cluster).
    
    Size fusion: Injects bounding box area ratio (25% weight) so different-sized 
    packages of the same brand get separated.
    
    Falls back to Agglomerative (complete linkage) if HDBSCAN is not installed.
    
    Returns: (unique_products, embeddings, cluster_labels)
    """
    if not crops_data:
        return [], np.array([]), np.array([])

    # Get embeddings for all crops
    embeddings = []
    areas = []
    for crop, bbox, conf in crops_data:
        emb = get_embedding(dinov2_model, crop)
        embeddings.append(emb)
        # Compute bounding box area for size-aware fusion
        x1, y1, x2, y2 = bbox
        areas.append((x2 - x1) * (y2 - y1))

    embeddings = np.array(embeddings, dtype=np.float32)
    areas = np.array(areas, dtype=np.float32)
    n = len(embeddings)

    if n == 1:
        cluster_labels = np.array([0])
        unique_products = [{
            "crop": crops_data[0][0],
            "bbox": crops_data[0][1],
            "confidence": crops_data[0][2],
            "count": 1,
            "embedding": embeddings[0].tolist(),
            "geometry": get_crop_geometry(crops_data[0][1]),
            "member_bboxes": [crops_data[0][1]],
        }]
        return unique_products, embeddings, cluster_labels

    # ── Build combined distance matrix (visual + size) ──
    # Visual distance: 1 - cosine_similarity
    visual_sim = np.dot(embeddings, embeddings.T)  # cosine sim (already L2-normalized)
    visual_dist = 1.0 - visual_sim

    # Size distance: 1 - min(a,b)/max(a,b) for each pair
    size_dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            size_ratio = min(areas[i], areas[j]) / max(areas[i], areas[j]) if max(areas[i], areas[j]) > 0 else 1.0
            sd = 1.0 - size_ratio
            size_dist[i, j] = sd
            size_dist[j, i] = sd

    # Combined distance: 75% visual + 25% size
    combined_dist = 0.75 * visual_dist + 0.25 * size_dist

    # ── Agglomerative Clustering with complete linkage ──
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Convert to condensed distance matrix for scipy
    condensed = squareform(combined_dist, checks=False)

    # Complete linkage: max distance between all pairs in two clusters
    Z = linkage(condensed, method='complete')

    # Cut tree at distance = (1 - similarity_threshold)
    distance_threshold = 1.0 - similarity_threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance') - 1  # 0-indexed

    # ── Build cluster groups ──
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    # Sort clusters by size (largest first)
    sorted_label_groups = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Re-label so that largest cluster = 0
    label_remap = {}
    for new_label, (old_label, members) in enumerate(sorted_label_groups):
        label_remap[old_label] = new_label
    cluster_labels = np.array([label_remap[l] for l in cluster_labels], dtype=int)

    # Build results with representative crop info
    unique_products = []
    for new_label, (old_label, members) in enumerate(sorted_label_groups):
        # Pick the highest-confidence crop as representative
        best_idx = max(members, key=lambda m: crops_data[m][2])
        crop, bbox, conf = crops_data[best_idx]
        member_bboxes = [crops_data[m][1] for m in members]
        unique_products.append({
            "crop": crop,
            "bbox": bbox,
            "confidence": conf,
            "count": len(members),
            "embedding": embeddings[best_idx].tolist(),
            "geometry": get_crop_geometry(bbox),
            "member_bboxes": member_bboxes,
        })

    return unique_products, embeddings, cluster_labels

# ── Product Catalog Management ────────────────────────────────────────────
def load_catalog():
    """Load product catalog from SQLite database."""
    return get_catalog_as_dict()

def save_catalog(catalog):
    """Save product catalog — handled by db.add_product() now."""
    pass  # Individual products are saved via db.add_product()

def build_faiss_index(catalog):
    """Build FAISS index from catalog embeddings."""
    import faiss
    embeddings = []
    valid_products = []
    for p in catalog["products"]:
        if "embedding" in p and p["embedding"] is not None:
            embeddings.append(p["embedding"])
            valid_products.append(p)
    if not embeddings:
        return None, []
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vectors)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, valid_products

def search_product(index, products, query_embedding, threshold=0.5):
    """Search FAISS index for matching product."""
    if index is None or len(products) == 0:
        return None, 0.0
    query = np.array([query_embedding], dtype=np.float32)
    scores, indices = index.search(query, 1)
    score = float(scores[0][0])
    if score >= threshold:
        return products[int(indices[0][0])], score
    return None, score


def search_product_with_size(index, products, query_embedding, query_bbox,
                              expected_products_on_shelf=None, threshold=0.5,
                              size_weight=0.15):
    """Enhanced search: DINOv2 visual similarity + bounding box size-ratio.
    
    Combines visual embedding score with height-ratio comparison
    to distinguish same-brand, different-size products.
    
    Args:
        index: FAISS index
        products: List of indexed products
        query_embedding: DINOv2 embedding of detected crop
        query_bbox: (x1, y1, x2, y2) of detected product
        expected_products_on_shelf: List of expected planogram positions with bbox info
        threshold: Minimum combined score to accept
        size_weight: How much size-ratio influences final score (0-1)
    
    Returns:
        (matched_product, combined_score)
    """
    if index is None or len(products) == 0:
        return None, 0.0

    # Step 1: Get top-3 visual matches from FAISS
    k = min(3, len(products))
    query = np.array([query_embedding], dtype=np.float32)
    scores, indices = index.search(query, k)

    if float(scores[0][0]) < threshold:
        return None, float(scores[0][0])

    # If no expected products or no bbox → fall back to pure visual
    if expected_products_on_shelf is None or query_bbox is None:
        return products[int(indices[0][0])], float(scores[0][0])

    # Step 2: Get detected product height
    qx1, qy1, qx2, qy2 = query_bbox
    query_height = qy2 - qy1

    # Step 3: Score each candidate with size-ratio fusion
    best_match = None
    best_combined = 0.0

    for rank in range(k):
        visual_score = float(scores[0][rank])
        if visual_score < threshold * 0.8:  # Skip very low visual matches
            continue
        candidate = products[int(indices[0][rank])]
        candidate_sku = candidate["sku"]

        # Find this SKU's expected bbox in planogram
        size_score = 1.0  # Default: no penalty
        for exp_prod in expected_products_on_shelf:
            if exp_prod.get("sku") == candidate_sku:
                exp_bbox = exp_prod.get("bbox")
                if exp_bbox and len(exp_bbox) == 4:
                    exp_height = exp_bbox[3] - exp_bbox[1]
                    if exp_height > 0 and query_height > 0:
                        # Height ratio — 1.0 = perfect match, <1.0 = size mismatch
                        height_ratio = min(query_height, exp_height) / max(query_height, exp_height)
                        size_score = height_ratio
                break

        # Fused score: visual * (1-w) + size * w
        combined = visual_score * (1 - size_weight) + size_score * size_weight

        if combined > best_combined:
            best_combined = combined
            best_match = candidate

    if best_match and best_combined >= threshold:
        return best_match, best_combined
    # Fallback to top visual match
    return products[int(indices[0][0])], float(scores[0][0])

# ── Planogram Management (now via SQLite) ─────────────────────────────────
def load_planograms():
    """Load all planograms from SQLite database."""
    return get_planograms()

def save_planogram(name, data):
    """Save planogram to SQLite database."""
    save_planogram_db(name, data)

# ── Alert Engine ──────────────────────────────────────────────────────────
def send_mobile_alert(title, message, priority="high"):
    """Send push notification via ntfy.sh."""
    topic = st.session_state.get("ntfy_topic", "shelfmind-alerts")
    try:
        # Remove emojis from title (HTTP headers must be latin-1 safe)
        clean_title = title.encode("ascii", "ignore").decode("ascii").strip()
        if not clean_title:
            clean_title = "ShelfMind Alert"

        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title": clean_title,
                "Priority": priority,
                "Tags": "warning" if priority == "high" else "loudspeaker",
            },
            timeout=5,
        )
        return True
    except Exception as e:
        return False

def format_compliance_alert(shelf_name, issues, compliance_pct):
    """Generate formatted compliance alert HTML."""
    if compliance_pct >= 90:
        css_class = "alert-ok"
        icon = "✅"
    elif compliance_pct >= 70:
        css_class = "alert-warning"
        icon = "⚠️"
    else:
        css_class = "alert-critical"
        icon = "🔴"

    issues_html = "<br>".join(f"• {issue}" for issue in issues)
    return f"""<div class="{css_class}">
        <strong>{icon} {shelf_name} — {compliance_pct:.0f}% Compliant</strong><br>
        {issues_html}
    </div>"""

# ── YOLO Detection Helper ────────────────────────────────────────────────
def run_detection(model, image, conf=0.25, max_det=200):
    """Run YOLO detection and return detections list."""
    results = model.predict(image, conf=conf, max_det=max_det, device="cpu", verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]) if box.cls is not None else 0,
            })
    return detections, results

def detect_shelf_levels(detections, img_height):
    """Auto-detect shelf levels from product Y-positions."""
    if len(detections) < 3:
        return [0.0, float(img_height)], 1

    y_centers = sorted([(d["bbox"][1] + d["bbox"][3]) / 2 for d in detections])
    gaps = [(y_centers[i] - y_centers[i-1], i) for i in range(1, len(y_centers))]

    # Minimum gap must be at least 15% of image height to count as a new shelf
    # This prevents splitting products on the same flat surface
    min_shelf_gap = img_height * 0.15

    # Also use statistical threshold: gap must be 3x the average small gap
    small_gaps = sorted([g for g, _ in gaps])[:max(len(gaps)//2, 1)]
    avg_small = sum(small_gaps) / len(small_gaps) if small_gaps else 10
    stat_threshold = avg_small * 3

    # Use the LARGER of the two thresholds
    sig_threshold = max(min_shelf_gap, stat_threshold)

    gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)
    top_gaps = [(g, idx) for g, idx in gaps_sorted if g > sig_threshold][:6]
    top_gaps_sorted = sorted(top_gaps, key=lambda x: y_centers[x[1]])

    boundaries = [0.0]
    for _, idx in top_gaps_sorted:
        boundaries.append((y_centers[idx-1] + y_centers[idx]) / 2)
    boundaries.append(float(img_height))

    return boundaries, len(boundaries) - 1

def assign_to_shelves(detections, boundaries):
    """Assign each detection to a shelf level."""
    shelf_assignments = {}
    for det in detections:
        y_center = (det["bbox"][1] + det["bbox"][3]) / 2
        for s in range(len(boundaries) - 1):
            if boundaries[s] <= y_center < boundaries[s + 1]:
                shelf_id = s + 1
                if shelf_id not in shelf_assignments:
                    shelf_assignments[shelf_id] = []
                # Sort by x position (left to right)
                det["shelf"] = shelf_id
                shelf_assignments[shelf_id].append(det)
                break
    # Sort products in each shelf by x-position
    for shelf_id in shelf_assignments:
        shelf_assignments[shelf_id].sort(key=lambda d: d["bbox"][0])
    return shelf_assignments

def draw_annotated_image(image, detections, product_labels=None):
    """Draw bounding boxes with confidence-colored labels on image."""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        match_score = det.get("match_score", det.get("confidence", 0))

        # Confidence-based coloring
        if det.get("status") == "unknown" or match_score < 0.3:
            color = "#ff4343"  # Red — unknown
        elif match_score >= 0.7:
            color = "#00d4aa"  # Green — high confidence
        elif match_score >= 0.5:
            color = "#ffaa00"  # Yellow — medium
        else:
            color = "#ff8c00"  # Orange — low

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        label = ""
        if product_labels and i < len(product_labels):
            label = product_labels[i]
        elif "product_name" in det:
            label = det["product_name"]
        else:
            label = f"P{i+1}"

        text = f"{label} ({match_score:.0%})" if label else f"{match_score:.0%}"

        bbox = draw.textbbox((x1, y1 - 18), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1 - 18), text, fill="#0a0a1a", font=font)

    return draw_img


# ══════════════════════════════════════════════════════════════════════════
# ── PHONE CAMERA CONFIG ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

def capture_from_phone(url, rotation=0):
    """Grab a single frame from IP Webcam stream with orientation fix."""
    import cv2
    from PIL import ImageOps
    try:
        # Use /shot.jpg for a single frame (more reliable than /video)
        shot_url = url.replace("/video", "/shot.jpg").replace("/videofeed", "/shot.jpg")
        if "/shot.jpg" not in shot_url:
            shot_url = url.rstrip("/") + "/shot.jpg"

        import urllib.request
        with urllib.request.urlopen(shot_url, timeout=5) as resp:
            arr = np.frombuffer(resp.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # Auto-fix EXIF orientation
                try:
                    pil_img = ImageOps.exif_transpose(pil_img)
                except Exception:
                    pass
                # Apply manual rotation if needed
                if rotation == 90:
                    pil_img = pil_img.rotate(-90, expand=True)
                elif rotation == 180:
                    pil_img = pil_img.rotate(180, expand=True)
                elif rotation == 270:
                    pil_img = pil_img.rotate(90, expand=True)
                return pil_img
    except Exception as e:
        st.error(f"❌ Cannot connect to phone camera: {e}")
    return None

# Phone camera global settings (collapsible)
with st.expander("📱 Phone Camera Setup", expanded=False):
    phone_cols = st.columns([2, 1, 1])
    with phone_cols[0]:
        phone_cam_url = st.text_input(
            "IP Webcam URL",
            value="http://192.168.1.5:8080",
            key="global_phone_url",
            help="Install 'IP Webcam' app on Android → Start Server → paste URL here"
        )
    with phone_cols[1]:
        phone_rotation = st.selectbox(
            "Rotation Fix",
            [0, 90, 180, 270],
            index=1,
            format_func=lambda x: f"🔄 {x}°" if x > 0 else "None",
            key="phone_rotation",
            help="If image appears sideways, change this"
        )
    with phone_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔗 Test Connection"):
            test_img = capture_from_phone(phone_cam_url, phone_rotation)
            if test_img:
                st.success("✅ Connected!")
                st.image(test_img, caption="Phone camera preview", width='stretch')
            else:
                st.error("❌ Cannot reach phone camera")

    st.markdown("""
    **Quick Setup:** Install **IP Webcam** (Android) → Start Server → Enter URL above  
    Both phone & laptop must be on the **same WiFi network**
    """)


# ══════════════════════════════════════════════════════════════════════════
# ── TABS ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📸 Product Scanner",
    "📋 Planogram Creator",
    "🎥 Live Monitor",
    "📊 Analytics",
    "📓 Training Results",
])



# ══════════════════════════════════════════════════════════════════════════
# ── TAB 1: PRODUCT SCANNER ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📸 Product Scanner — Register Your Products</div>', unsafe_allow_html=True)
    st.caption("Register products individually or bulk-scan a store shelf image to auto-detect and register all unique products.")

    # ── Choose Scan Mode ──────────────────────────────────────────────
    scan_mode = st.radio(
        "Scan Mode",
        ["📷 Single Product Scan", "🏪 Bulk Store Shelf Scan"],
        horizontal=True, key="scan_mode"
    )

    # ══════════════════════════════════════════════════════════════════
    # ── MODE 1: SINGLE PRODUCT SCAN (with auto-crop + OCR) ──────────
    # ══════════════════════════════════════════════════════════════════
    if scan_mode == "📷 Single Product Scan":
        catalog = load_catalog()

        # ── Model & Processing Toggles ──
        toggle_cols = st.columns(2)
        with toggle_cols[0]:
            dinov2_choice = st.radio(
                "🧠 DINOv2 Model",
                ["Pretrained (ViT-S/14)", "Fine-tuned (ViT-B/14)"],
                horizontal=True, key="dinov2_choice",
                help="Compare pretrained vs fine-tuned DINOv2 embeddings"
            )
        with toggle_cols[1]:
            use_rembg = st.checkbox("✂️ Use rembg (background removal)", value=True, key="use_rembg",
                                   help="Toggle AI background removal before embedding")

        col_cam, col_form = st.columns([1, 1])

        with col_cam:
            st.markdown("##### Capture Product Image")
            img_source = st.radio(
                "Image Source",
                ["📱 Phone Camera", "💻 Laptop Camera", "📁 Upload"],
                horizontal=True, key="scanner_source"
            )

            captured_img = None
            if img_source == "📱 Phone Camera":
                st.info("Point your phone at the product, then click capture 👇")
                if st.button("📸 Capture from Phone", type="primary", key="phone_cap_scanner"):
                    captured_img = capture_from_phone(phone_cam_url, phone_rotation)
                    if captured_img:
                        st.session_state["scanner_phone_img"] = captured_img
                # Persist across reruns
                if "scanner_phone_img" in st.session_state:
                    captured_img = st.session_state["scanner_phone_img"]

            elif img_source == "💻 Laptop Camera":
                camera_photo = st.camera_input("Point camera at the product", key="scanner_cam")
                if camera_photo:
                    captured_img = Image.open(camera_photo).convert("RGB")
                    st.session_state["scanner_captured_img"] = captured_img
                elif "scanner_captured_img" in st.session_state:
                    captured_img = st.session_state["scanner_captured_img"]
            else:
                uploaded_photo = st.file_uploader(
                    "Upload product photo",
                    type=["jpg", "jpeg", "png"],
                    key="scanner_upload"
                )
                if uploaded_photo:
                    captured_img = Image.open(uploaded_photo).convert("RGB")
                    st.session_state["scanner_captured_img"] = captured_img
                elif "scanner_captured_img" in st.session_state:
                    captured_img = st.session_state["scanner_captured_img"]

            # Show captured image + barcode + optional OCR
            if captured_img:
                st.image(captured_img, caption="Captured product", width='stretch')
                st.caption("💡 *Tip: Frame the product close-up with the label/barcode facing the camera*")

                # Check if we need to re-process (new image vs previously processed)
                import hashlib
                img_hash = hashlib.md5(np.array(captured_img).tobytes()[:10000]).hexdigest()
                needs_processing = st.session_state.get("scanner_img_hash") != img_hash

                if needs_processing:
                    st.session_state["scanner_img_hash"] = img_hash

                    # Auto-scan barcode + lookup product info
                    barcode_text = scan_barcode(captured_img)
                    barcode_info = {}
                    if barcode_text:
                        st.success(f"📊 **Barcode detected:** `{barcode_text}`")
                        with st.spinner("Looking up product info from barcode..."):
                            barcode_info = lookup_barcode_info(barcode_text)
                        if barcode_info.get("name"):
                            st.info(f"🏷️ **Auto-identified:** {barcode_info['name']}")
                            if barcode_info.get("category"):
                                st.caption(f"Category: {barcode_info['category']}")
                        else:
                            st.caption("Product not found in online database — enter name manually")
                    else:
                        st.caption("No barcode found (flip product to show barcode, or enter name manually)")

                    # Optional OCR for name suggestion
                    ocr_text = ""
                    if st.checkbox("🔤 Run OCR to read label text", value=False, key="run_ocr"):
                        ocr_reader = load_ocr()
                        if ocr_reader:
                            with st.spinner("Reading label text..."):
                                ocr_text = extract_text_from_crop(ocr_reader, captured_img)
                            if ocr_text:
                                st.info(f"📝 **OCR detected:** {ocr_text}")
                            else:
                                st.caption("No readable text found on label")

                    # ── rembg Background Removal (conditional) ──
                    import cv2
                    cropped_img = captured_img  # fallback to full image
                    if use_rembg:
                        try:
                            from rembg import remove
                            with st.spinner("✂️ Removing background (AI segmentation)..."):
                                result = remove(captured_img)
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
                                    cropped_img = captured_img.crop((x1, y1, x2, y2))
                                    st.success("✂️ Product perfectly cropped (AI background removal)")
                                    st.image(cropped_img, caption="Auto-cropped product (display only — embedding uses original)", width=250)
                                else:
                                    st.caption("ℹ️ Using full image")
                        except ImportError:
                            st.caption("ℹ️ rembg not installed — using full image")
                        except Exception as e:
                            st.caption(f"ℹ️ Using full image (crop error: {e})")
                    else:
                        st.caption("ℹ️ rembg OFF — using full image (no background removal)")

                    # Store for form submission
                    st.session_state["scanner_cropped"] = cropped_img
                    st.session_state["scanner_full_img"] = captured_img
                    auto_name = barcode_info.get("name", "") or ocr_text
                    st.session_state["scanner_ocr_text"] = auto_name
                    st.session_state["scanner_barcode"] = barcode_text or ""
                    st.session_state["scanner_barcode_info"] = barcode_info

                    # ── Augmented Views Grid ──
                    dinov2_model = load_dinov2_finetuned() if dinov2_choice == "Fine-tuned (ViT-B/14)" else load_dinov2()
                    if dinov2_model:
                        with st.spinner(f"Generating 15 augmented views ({dinov2_choice})..."):
                            # Always embed from ORIGINAL image (not rembg crop)
                            # rembg removes background → domain mismatch with shelf queries → lower scores
                            result = get_robust_embedding(dinov2_model, captured_img, return_views=True)
                            emb_vec, aug_views, view_names = result
                            st.session_state["scanner_aug_views"] = aug_views
                            st.session_state["scanner_view_names"] = view_names
                            st.session_state["scanner_embedding"] = emb_vec

                else:
                    # Restore cached results (no re-processing needed)
                    cropped_img = st.session_state.get("scanner_cropped", captured_img)
                    if st.session_state.get("scanner_barcode"):
                        st.success(f"📊 **Barcode:** `{st.session_state['scanner_barcode']}`")
                    if st.session_state.get("scanner_ocr_text"):
                        st.info(f"🏷️ **Product:** {st.session_state['scanner_ocr_text']}")
                    if cropped_img != captured_img:
                        st.image(cropped_img, caption="Auto-cropped product (display only)", width=250)

                # Display augmented views (from cache or fresh)
                aug_views = st.session_state.get("scanner_aug_views")
                view_names = st.session_state.get("scanner_view_names")
                if aug_views and view_names:
                    st.markdown(f"**🔄 {len(aug_views)} Augmented Views ({dinov2_choice}):**")
                    for row_start in range(0, len(aug_views), 5):
                        row = st.columns(5)
                        for i in range(row_start, min(row_start + 5, len(aug_views))):
                            with row[i - row_start]:
                                st.image(aug_views[i], caption=view_names[i], width="content")

        with col_form:
            st.markdown("##### Product Details")
            # Pre-fill name from OCR/barcode/voice if available
            default_name = st.session_state.get("scanner_ocr_text", "")
            default_barcode = st.session_state.get("scanner_barcode", "")

            # Voice input via Web Speech API — auto-fills Name, Price, Category
            import streamlit.components.v1 as components
            components.html("""
            <div style="margin-bottom:4px; font-family: 'Source Sans Pro', sans-serif;">
                <button id="voiceBtn" onclick="startVoice()" style="
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    color: white; border: none; padding: 8px 16px;
                    border-radius: 8px; cursor: pointer; font-size: 14px;
                    display: inline-flex; align-items: center; gap: 6px;
                ">🎤 Speak Product Details</button>
                <span id="voiceStatus" style="color:#aaa; font-size:13px; margin-left:8px;">
                    Say: "Coca Cola 330ml price 40 rupees"
                </span>
            </div>
            <script>
            // Helper: set React-controlled input value
            function setInput(input, value) {
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(input, value);
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }

            // Helper: find Streamlit input by placeholder or label text
            function findInput(searchText) {
                const doc = window.parent.document;
                // Try by placeholder (text inputs)
                let el = doc.querySelector('input[placeholder*="' + searchText + '"]');
                if (el) return el;
                // Try by label — walk up multiple parent levels to find the input
                const labels = doc.querySelectorAll('label');
                for (const lbl of labels) {
                    if (lbl.textContent.includes(searchText)) {
                        // Search in progressively larger parent containers
                        let container = lbl.parentElement;
                        for (let depth = 0; depth < 5 && container; depth++) {
                            el = container.querySelector('input[type="text"], input[type="number"], input:not([type])');
                            if (el) return el;
                            container = container.parentElement;
                        }
                    }
                }
                return null;
            }

            // Voice only handles name + price. Category is auto-detected
            // from the product IMAGE using FAISS (Python-side, see below).

            function startVoice() {
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                    document.getElementById('voiceStatus').innerText = '❌ Speech not supported';
                    return;
                }
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                recognition.lang = 'en-IN';
                recognition.continuous = false;
                recognition.interimResults = false;
                document.getElementById('voiceBtn').style.background = '#ef4444';
                document.getElementById('voiceBtn').innerHTML = '🔴 Listening...';
                document.getElementById('voiceStatus').innerText = 'Speak now...';
                recognition.start();
                recognition.onresult = function(event) {
                    let text = event.results[0][0].transcript;
                    document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
                    document.getElementById('voiceBtn').innerHTML = '🎤 Speak Product Details';

                    // Extract price — handles speech variants: rupees, ruppees, rupay, rs, price
                    let price = '';
                    const pricePatterns = [
                        /(?:price|₹|rs\.?)\s*(\d+\.?\d*)/i,
                        /(\d+\.?\d*)\s*(?:rupees?|ruppees?|rupay|rupaiye?|rs\.?)/i,
                        /(?:rupees?|ruppees?|rupay|rupaiye?)\s*(\d+\.?\d*)/i,
                    ];
                    for (const pattern of pricePatterns) {
                        const m = text.match(pattern);
                        if (m) {
                            price = m[1];
                            text = text.replace(m[0], '').trim();
                            break;
                        }
                    }

                    // Clean up name — remove filler words
                    let productName = text.replace(/^\s*(the|a|an|its?|and)\s+/i, '').trim();
                    productName = productName.replace(/\s+/g, ' ').trim();
                    if (productName.length > 0) {
                        productName = productName.charAt(0).toUpperCase() + productName.slice(1);
                    }

                    document.getElementById('voiceStatus').innerText = '✅ ' + productName + (price ? ' | ₹' + price : '');

                    // Auto-fill Product Name
                    const nameInput = findInput('Coca-Cola');
                    if (nameInput) setInput(nameInput, productName);

                    // Auto-fill Price (number input)
                    if (price) {
                        const priceInput = findInput('Price');
                        if (priceInput) {
                            setInput(priceInput, price);
                            // Also dispatch for React number input
                            priceInput.dispatchEvent(new Event('blur', { bubbles: true }));
                        }
                    }

                    // Category is auto-detected from the product IMAGE
                    // via FAISS on the Python side (not from voice text)
                };
                recognition.onerror = function(e) {
                    document.getElementById('voiceStatus').innerText = '❌ ' + e.error;
                    document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
                    document.getElementById('voiceBtn').innerHTML = '🎤 Speak Product Details';
                };
            }
            </script>
            """, height=50)

            # ── Auto-detect category from product image via FAISS ──
            categories = [
                "Beverages", "Snacks", "Dairy", "Canned Goods",
                "Bakery", "Cleaning", "Personal Care", "Frozen",
                "Fruits & Vegetables", "Other"
            ]
            suggested_category_idx = len(categories) - 1  # Default: "Other"
            if "scanner_embedding" in st.session_state:
                try:
                    catalog = load_catalog()
                    faiss_idx, faiss_prods = build_faiss_index(catalog)
                    if faiss_idx and faiss_prods:
                        query_emb = np.array([st.session_state["scanner_embedding"]], dtype=np.float32)
                        scores, indices = faiss_idx.search(query_emb, 1)
                        if float(scores[0][0]) >= 0.5:
                            matched_cat = faiss_prods[int(indices[0][0])].get("category", "Other")
                            if matched_cat in categories:
                                suggested_category_idx = categories.index(matched_cat)
                                st.caption(f"🧠 Category auto-suggested from similar product (similarity: {float(scores[0][0]):.2f})")
                except Exception:
                    pass

            with st.form("product_form", clear_on_submit=True):
                prod_name = st.text_input("Product Name *", value=default_name, placeholder="e.g., Coca-Cola 330ml (or use 🎤 above)")
                prod_barcode = st.text_input("Barcode", value=default_barcode, placeholder="Auto-detected or enter manually")
                p_cols = st.columns(2)
                with p_cols[0]:
                    prod_price = st.number_input("Price (₹)", min_value=0.0, value=0.0, step=0.5)
                with p_cols[1]:
                    prod_category = st.selectbox("Category", categories, index=suggested_category_idx)

                submitted = st.form_submit_button("✅ Register Product", type="primary", width='stretch')

                if submitted and prod_name:
                    # Use auto-cropped image if available
                    reg_img = st.session_state.get("scanner_cropped", captured_img)
                    if reg_img:
                        with st.spinner("Registering product..."):
                            next_id = get_next_product_id()
                            sku_id = f"SKU_{next_id:04d}"

                            # Save product image
                            img_filename = f"{sku_id}_{re.sub(r'[^a-z0-9_]', '', prod_name.replace(' ', '_').lower())}.jpg"
                            img_path = REF_IMG_DIR / img_filename
                            reg_img.save(str(img_path), "JPEG", quality=90)

                            # Use pre-computed embedding from augmented views grid
                            embedding = None
                            if "scanner_embedding" in st.session_state:
                                embedding = st.session_state["scanner_embedding"].tolist()
                            else:
                                dinov2 = load_dinov2_finetuned() if dinov2_choice == "Fine-tuned (ViT-B/14)" else load_dinov2()
                                if dinov2:
                                    emb_vec = get_robust_embedding(dinov2, reg_img)
                                    embedding = emb_vec.tolist()

                            # Save to SQLite database
                            add_product(
                                sku=sku_id,
                                name=prod_name,
                                category=prod_category,
                                price=prod_price,
                                image_path=img_filename,
                                embedding=embedding,
                                barcode=prod_barcode if prod_barcode else None,
                            )

                            # Clear augmented views from session
                            for key in ["scanner_aug_views", "scanner_view_names", "scanner_embedding",
                                        "scanner_cropped", "scanner_full_img", "scanner_ocr_text", "scanner_barcode"]:
                                st.session_state.pop(key, None)

                            st.success(f"✅ **{prod_name}** registered as **{sku_id}** in database!")
                            st.rerun()
                    else:
                        st.warning("Please capture a product photo first.")
                elif submitted and not prod_name:
                    st.warning("Please enter a product name.")

    # ══════════════════════════════════════════════════════════════════
    # ── MODE 2: BULK STORE SHELF SCAN ────────────────────────────────
    # ══════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        **How it works:**  
        1️⃣ Capture/upload a store shelf image → 2️⃣ Detector finds all products → 3️⃣ DINOv2 clusters unique ones →  
        4️⃣ OCR reads text from each → 5️⃣ Label & register all unique products at once!
        """)

        # ── Model Selector ──
        toggle_cols2 = st.columns(2)
        with toggle_cols2[0]:
            det_model_choice = st.radio(
                "🔧 Detection Model",
                ["⚡ YOLO26s (Fine-tuned)", "🎯 RF-DETR (Fine-tuned)"],
                horizontal=True, key="det_model_choice",
                help="Compare detection accuracy between YOLO26s and RF-DETR on the same shelf image"
            )
        with toggle_cols2[1]:
            bulk_dinov2_choice = st.radio(
                "🧠 DINOv2 Model",
                ["Pretrained (ViT-S/14)", "Fine-tuned (ViT-B/14)"],
                horizontal=True, key="bulk_dinov2_choice",
                help="Compare pretrained vs fine-tuned DINOv2 for clustering"
            )

        # ── Shelf Image Source (Phone / Laptop Camera / Upload) ──
        shelf_source = st.radio(
            "Image Source",
            ["📱 Phone Camera", "💻 Laptop Camera", "📁 Upload"],
            horizontal=True, key="bulk_shelf_source"
        )

        shelf_img = None
        if shelf_source == "📱 Phone Camera":
            st.info("Point your phone at the store shelf, then click capture 👇")
            if st.button("📸 Capture Shelf from Phone", type="primary", key="phone_cap_bulk"):
                phone_img = capture_from_phone(phone_cam_url, phone_rotation)
                if phone_img:
                    st.session_state["bulk_shelf_phone_img"] = phone_img
            if "bulk_shelf_phone_img" in st.session_state:
                shelf_img = st.session_state["bulk_shelf_phone_img"]

        elif shelf_source == "💻 Laptop Camera":
            camera_photo = st.camera_input("Point camera at the store shelf", key="bulk_shelf_cam")
            if camera_photo:
                shelf_img = Image.open(camera_photo).convert("RGB")
                st.session_state["bulk_shelf_cam_img"] = shelf_img
            elif "bulk_shelf_cam_img" in st.session_state:
                shelf_img = st.session_state["bulk_shelf_cam_img"]

        else:
            shelf_upload = st.file_uploader(
                "📤 Upload Store Shelf Image",
                type=["jpg", "jpeg", "png"],
                key="bulk_shelf_upload"
            )
            if shelf_upload:
                shelf_img = Image.open(shelf_upload).convert("RGB")
                st.session_state["bulk_shelf_upload_img"] = shelf_img
            elif "bulk_shelf_upload_img" in st.session_state:
                shelf_img = st.session_state["bulk_shelf_upload_img"]

        if shelf_img:
            # Detect if shelf image changed — clear stale results
            import hashlib
            shelf_hash = hashlib.md5(np.array(shelf_img).tobytes()[:10000]).hexdigest()
            if st.session_state.get("bulk_shelf_hash") != shelf_hash:
                st.session_state["bulk_shelf_hash"] = shelf_hash
                # Clear old results when image changes
                st.session_state.pop("bulk_unique_products", None)
                st.session_state.pop("bulk_shelf_img", None)

            st.image(shelf_img, caption=f"Shelf image ({shelf_img.size[0]}×{shelf_img.size[1]})", width='stretch')

            # ── Detection Confidence Threshold ──
            bulk_conf = st.slider(
                "Detection Confidence Threshold", 0.15, 0.90, 0.35, 0.05,
                key="bulk_conf_threshold",
                help="Higher = fewer false positives (shadows, reflections). Lower = catches more products but may include noise."
            )

            # ── Clustering Similarity Threshold ──
            cluster_thresh = st.slider(
                "DINOv2 Clustering Similarity Threshold", 0.60, 0.95, 0.82, 0.01,
                key="bulk_cluster_threshold",
                help="Higher = more unique products (stricter grouping). Lower = fewer unique products (merges similar ones). Try 0.78-0.85 for best results."
            )

            # Detection + Clustering
            if st.button("🔍 Detect & Extract Unique Products", type="primary", key="bulk_detect"):
                # Load selected detection model
                if det_model_choice == "🎯 RF-DETR (Fine-tuned)":
                    det_model = load_rfdetr()
                    model_name = "RF-DETR"
                else:
                    det_model = load_yolo()
                    model_name = "YOLO26s"
                dinov2 = load_dinov2_finetuned() if bulk_dinov2_choice == "Fine-tuned (ViT-B/14)" else load_dinov2()
                dinov2_label = "Fine-tuned + Projector (256-dim)" if bulk_dinov2_choice == "Fine-tuned (ViT-B/14)" else "Pretrained (384-dim)"

                if det_model and dinov2:
                    with st.spinner(f"Step 1/2: Detecting products with {model_name}..."):
                        raw_crops = detect_all_products(shelf_img, det_model, conf=bulk_conf)
                        raw_count = len(raw_crops)

                        # ── Filter false positives (shadows, reflections, non-products) ──
                        img_w, img_h = shelf_img.size
                        img_area = img_w * img_h
                        min_crop_area = img_area * 0.001  # Must be at least 0.1% of image area
                        max_crop_area = img_area * 0.25   # Can't be more than 25% of image
                        min_dimension = 20  # Minimum 20px in both width and height

                        all_crops = []
                        filtered_count = 0
                        for crop, bbox, conf_score in raw_crops:
                            bx1, by1, bx2, by2 = bbox
                            crop_w = bx2 - bx1
                            crop_h = by2 - by1
                            crop_area = crop_w * crop_h

                            # Filter 1: Too small (shadows, noise, floor reflections)
                            if crop_area < min_crop_area or crop_w < min_dimension or crop_h < min_dimension:
                                filtered_count += 1
                                continue

                            # Filter 2: Too large (entire shelf detected as one object)
                            if crop_area > max_crop_area:
                                filtered_count += 1
                                continue

                            # Filter 3: Extreme aspect ratio (tube lights, shelf edges, signage)
                            aspect = max(crop_w, crop_h) / max(min(crop_w, crop_h), 1)
                            if aspect > 6.0:  # Products rarely have >6:1 aspect ratio
                                filtered_count += 1
                                continue

                            all_crops.append((crop, bbox, conf_score))

                        if filtered_count > 0:
                            st.caption(f"🧹 Filtered {filtered_count} false positives (shadows, reflections, non-products)")
                        st.info(f"🔍 **{model_name}** detected **{len(all_crops)}** valid products (from {raw_count} raw detections)")

                        # Draw all detections on image
                        annotated = shelf_img.copy()
                        draw = ImageDraw.Draw(annotated)
                        for crop, bbox, conf_score in all_crops:
                            draw.rectangle(bbox, outline="lime", width=8)
                        st.image(annotated, caption=f"{model_name} — Filtered detections ({len(all_crops)} products)", width='stretch')

                    if all_crops:
                        with st.spinner(f"Step 2/2: Clustering unique products with DINOv2 {dinov2_label}..."):
                            unique, all_embeddings, cluster_labels = cluster_unique_products(all_crops, dinov2, similarity_threshold=cluster_thresh)
                            st.success(f"✅ Found **{len(unique)}** unique product types from {len(all_crops)} detections")

                        # OCR disabled — user types product names manually during registration
                        for prod in unique:
                            prod["ocr_text"] = ""

                        # ── Draw labeled annotated image with product names ──
                        import colorsys
                        def generate_distinct_colors(n):
                            """Generate N visually distinct colors using HSV space."""
                            colors = []
                            for i in range(n):
                                hue = i / max(n, 1)  # evenly spaced hues
                                sat = 0.9 + (i % 2) * 0.1  # high saturation for vivid colors
                                val = 1.0  # full brightness for maximum visibility
                                r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 1.0), val)
                                colors.append(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
                            return colors

                        label_colors = generate_distinct_colors(len(unique))
                        labeled_img = shelf_img.copy()
                        labeled_draw = ImageDraw.Draw(labeled_img)
                        # Try to load a readable font — scale by image width
                        try:
                            from PIL import ImageFont
                            img_w = shelf_img.width
                            font_size = max(20, int(img_w / 60))  # ~35px on 2000px wide image
                            font = ImageFont.truetype("arial.ttf", font_size)
                            font_small = ImageFont.truetype("arialbd.ttf", font_size)  # bold for labels
                        except Exception:
                            try:
                                font_small = ImageFont.truetype("arial.ttf", font_size)
                            except Exception:
                                font_small = ImageFont.load_default()
                            font = font_small

                        for i, prod in enumerate(unique):
                            color = label_colors[i % len(label_colors)]
                            label_name = prod.get("ocr_text", "")[:25] or f"Product_{i+1}"
                            prod["display_label"] = label_name  # Store for later use

                            for bbox in prod.get("member_bboxes", [prod["bbox"]]):
                                x1, y1, x2, y2 = bbox
                                labeled_draw.rectangle(bbox, outline=color, width=8)
                                # Draw label background above the box
                                tag = f"#{i+1} {label_name}"
                                text_bbox = labeled_draw.textbbox((x1, y1), tag, font=font_small)
                                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                                label_y = max(0, y1 - th - 10)
                                labeled_draw.rectangle([x1, label_y, x1 + tw + 12, label_y + th + 8], fill=color)
                                labeled_draw.text((x1 + 6, label_y + 4), tag, fill="white", font=font_small)

                        st.image(labeled_img, caption=f"Clustered — {len(unique)} unique products labeled", width='stretch')

                        # ── t-SNE Embedding Visualization ──────────────────────
                        st.markdown("---")
                        st.markdown("##### 🧬 DINOv2 Embedding Space — t-SNE Visualization")
                        st.caption("Each dot = one detected product crop. Color = cluster assignment. Products in the same cluster share the same color.")

                        try:
                            from sklearn.manifold import TSNE
                            import plotly.graph_objects as go

                            n_samples = len(all_embeddings)
                            perplexity = min(30, max(2, n_samples - 1))

                            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
                            coords_2d = tsne.fit_transform(all_embeddings)

                            # Build cluster display names
                            cluster_names = []
                            for lbl in cluster_labels:
                                if lbl < len(unique):
                                    name = unique[lbl].get("display_label", f"Product_{lbl+1}")
                                else:
                                    name = f"Product_{lbl+1}"
                                cluster_names.append(f"#{lbl+1} {name}")

                            # Generate plotly colors matching label_colors
                            n_unique = len(unique)
                            plotly_colors = []
                            for i in range(n_unique):
                                hue = i / max(n_unique, 1)
                                sat = 0.9 + (i % 2) * 0.1
                                r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 1.0), 1.0)
                                plotly_colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")

                            fig = go.Figure()
                            for cluster_id in range(n_unique):
                                mask = cluster_labels == cluster_id
                                if not np.any(mask):
                                    continue
                                count = int(np.sum(mask))
                                name = unique[cluster_id].get("display_label", f"Product_{cluster_id+1}")
                                fig.add_trace(go.Scatter(
                                    x=coords_2d[mask, 0],
                                    y=coords_2d[mask, 1],
                                    mode='markers',
                                    marker=dict(
                                        size=12,
                                        color=plotly_colors[cluster_id % len(plotly_colors)],
                                        line=dict(width=1, color='white'),
                                        opacity=0.9
                                    ),
                                    name=f"#{cluster_id+1} {name} (×{count})",
                                    hovertemplate=f"#{cluster_id+1} {name}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>"
                                ))

                            fig.update_layout(
                                title=dict(
                                    text=f"DINOv2 Embedding Clusters — {n_samples} crops → {n_unique} unique products",
                                    font=dict(size=14, color='#ccd6f6')
                                ),
                                xaxis_title="t-SNE Dimension 1",
                                yaxis_title="t-SNE Dimension 2",
                                plot_bgcolor='#0a192f',
                                paper_bgcolor='#0a192f',
                                font=dict(color='#8892b0'),
                                legend=dict(
                                    bgcolor='rgba(10,25,47,0.8)',
                                    bordercolor='#1e3a5f',
                                    borderwidth=1,
                                    font=dict(size=11)
                                ),
                                xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
                                yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
                                height=500,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        except ImportError:
                            st.info("Install scikit-learn and plotly for embedding visualization: `pip install scikit-learn plotly`")
                        except Exception as e:
                            st.warning(f"t-SNE visualization error: {e}")

                        # Store in session state
                        st.session_state["bulk_unique_products"] = unique
                        st.session_state["bulk_shelf_img"] = shelf_img
                    else:
                        st.warning("No products detected. Try an image with more visible products.")
                else:
                    st.error(f"{model_name} or DINOv2 not loaded. Check model files.")

        # ── Display unique products for labeling ──────────────────────
        if "bulk_unique_products" in st.session_state:
            unique_products = st.session_state["bulk_unique_products"]
            st.markdown("---")
            st.markdown(f"##### 🏷️ Label & Register Unique Products ({len(unique_products)} found)")
            st.caption("Review each unique product, edit the auto-suggested name, and register them all.")

            # Collect labels
            product_labels = []
            for i, prod in enumerate(unique_products):
                with st.container():
                    cols = st.columns([1, 2, 1, 1, 1])
                    with cols[0]:
                        st.image(prod["crop"], width=100)
                    with cols[1]:
                        # Auto-suggest name from OCR
                        suggested = prod.get("ocr_text", "")[:50] if prod.get("ocr_text") else f"Product_{i+1}"
                        if not suggested.strip():
                            suggested = f"Product_{i+1}"
                        name = st.text_input(
                            f"Name",
                            value=suggested,
                            key=f"bulk_name_{i}",
                            label_visibility="collapsed",
                            placeholder="Product name"
                        )
                        if prod.get("ocr_text"):
                            st.caption(f"📝 OCR: {prod['ocr_text'][:80]}")
                    with cols[2]:
                        category = st.selectbox("Category", [
                            "Beverages", "Snacks", "Dairy", "Canned Goods",
                            "Bakery", "Cleaning", "Personal Care", "Frozen",
                            "Fruits & Vegetables", "Other"
                        ], key=f"bulk_cat_{i}", label_visibility="collapsed")
                    with cols[3]:
                        geo = prod.get("geometry", {})
                        st.caption(f"📐 {geo.get('width', 0)}×{geo.get('height', 0)}")
                        st.caption(f"🔢 ×{prod['count']} instances")
                    with cols[4]:
                        include = st.checkbox("Register", value=True, key=f"bulk_inc_{i}")

                    product_labels.append({
                        "name": name, "category": category,
                        "include": include, "idx": i
                    })

            # Register all button
            st.markdown("---")
            if st.button("✅ Register All Selected Products", type="primary", key="bulk_register"):
                dinov2 = load_dinov2()
                registered_count = 0
                progress = st.progress(0, text="Registering products...")

                selected = [p for p in product_labels if p["include"] and p["name"].strip()]
                for j, label in enumerate(selected):
                    prod = unique_products[label["idx"]]
                    next_id = get_next_product_id()
                    sku_id = f"SKU_{next_id:04d}"

                    # Save crop image
                    img_filename = f"{sku_id}_{re.sub(r'[^a-z0-9_]', '', label['name'].replace(' ', '_').lower())}.jpg"
                    img_path = REF_IMG_DIR / img_filename
                    prod["crop"].save(str(img_path), "JPEG", quality=90)

                    # Compute robust 15-view averaged embedding for registration
                    progress.progress((j + 0.5) / len(selected), text=f"Computing robust embedding for {label['name']}...")
                    if dinov2:
                        embedding = get_robust_embedding(dinov2, prod["crop"]).tolist()
                    else:
                        embedding = prod.get("embedding")  # fallback to single-view

                    # Save to database
                    add_product(
                        sku=sku_id,
                        name=label["name"],
                        category=label["category"],
                        price=0.0,
                        image_path=img_filename,
                        embedding=embedding,
                    )
                    registered_count += 1
                    progress.progress((j + 1) / len(selected), text=f"Registered {label['name']}...")

                progress.empty()
                st.success(f"✅ Registered **{registered_count}** products from store shelf!")
                # Clear session state
                if "bulk_unique_products" in st.session_state:
                    del st.session_state["bulk_unique_products"]
                st.rerun()

            # ── FAISS Similarity Visualization ─────────────────────────────
            st.markdown("---")
            st.markdown("##### 🔍 FAISS Embedding Similarity Analysis")
            st.caption("Each detected unique product is queried against the catalog. Shows how well DINOv2 embeddings differentiate products.")

            catalog = load_catalog()
            if len(catalog["products"]) >= 2:
                import faiss
                faiss_index, faiss_products = build_faiss_index(catalog)
                if faiss_index and faiss_products:
                    # Determine index dimension for compatibility check
                    index_dim = faiss_index.d

                    # Load matching DINOv2 model if needed for re-embedding
                    reembed_model = None  # Will be loaded lazily on dimension mismatch

                    for i, prod in enumerate(unique_products):
                        emb = prod.get("embedding")
                        if emb is None:
                            continue

                        emb = np.array(emb, dtype=np.float32)

                        # ── Dimension mismatch guard ──
                        if emb.shape[0] != index_dim:
                            # Re-embed the crop with the model that matches the catalog
                            if reembed_model is None:
                                if index_dim == 256:
                                    reembed_model = load_dinov2_finetuned()
                                else:
                                    reembed_model = load_dinov2()
                            if reembed_model is not None:
                                emb = get_embedding(reembed_model, prod["crop"])
                                emb = np.array(emb, dtype=np.float32)
                            else:
                                continue  # Skip if we can't match dimensions

                        label_name = prod.get("display_label", prod.get("ocr_text", "")[:25] or f"Product_{i+1}")

                        # Query FAISS for top-3 matches
                        query = np.array([emb], dtype=np.float32)
                        k = min(3, len(faiss_products))
                        scores, indices = faiss_index.search(query, k)

                        with st.container():
                            st.markdown(f"**#{i+1} {label_name}** (×{prod['count']} instances)")
                            match_cols = st.columns([1] + [1] * k + [1])

                            # Query image
                            with match_cols[0]:
                                st.image(prod["crop"], caption="🔎 Query", width=100)

                            # Top matches from catalog
                            for m in range(k):
                                idx = int(indices[0][m])
                                score = float(scores[0][m])
                                matched = faiss_products[idx]
                                with match_cols[m + 1]:
                                    # Try to load catalog image
                                    cat_img_path = REF_IMG_DIR / matched.get("image_path", "")
                                    if cat_img_path.exists():
                                        cat_img = Image.open(str(cat_img_path)).convert("RGB")
                                        st.image(cat_img, width=100)
                                    else:
                                        st.caption("🖼️ No image")
                                    # Color code similarity
                                    if score >= 0.85:
                                        badge = f"🟢 {score:.3f}"
                                    elif score >= 0.65:
                                        badge = f"🟡 {score:.3f}"
                                    else:
                                        badge = f"🔴 {score:.3f}"
                                    st.caption(f"{badge}\n{matched.get('name', 'Unknown')[:20]}")

                            with match_cols[-1]:
                                best_score = float(scores[0][0])
                                if best_score >= 0.85:
                                    st.success("✅ Match")
                                elif best_score >= 0.65:
                                    st.warning("⚠️ Weak")
                                else:
                                    st.error("❌ New")
                            st.markdown("---")
                else:
                    st.info("ℹ️ FAISS index empty — register products first to see similarity analysis.")
            else:
                st.info("ℹ️ Need 2+ registered products in catalog to run FAISS analysis.")

    # ── Product Gallery (shared between both modes) ───────────────────
    st.markdown("---")
    catalog = load_catalog()
    n_products = len(catalog["products"])
    has_embeddings = sum(1 for p in catalog["products"] if p.get("embedding"))

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_products}</div>
            <div class="metric-label">Products Registered</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{has_embeddings}</div>
            <div class="metric-label">Embeddings Ready</div>
        </div>""", unsafe_allow_html=True)
    with g3:
        status = "✅ Ready" if has_embeddings >= 2 else "⏳ Need 2+ products"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.4rem;">{status}</div>
            <div class="metric-label">FAISS Index Status</div>
        </div>""", unsafe_allow_html=True)

    if catalog["products"]:
        st.markdown("##### 🗄️ Registered Products")

        # Select all / Clear all / Delete buttons
        btn_cols = st.columns([1, 1, 1, 2])
        with btn_cols[0]:
            select_all = st.button("☑️ Select All")
        with btn_cols[1]:
            clear_sel = st.button("⬜ Clear Selection")
        with btn_cols[2]:
            if st.button("🗑️ Clear ALL", type="secondary"):
                clear_all_products()
                for f in REF_IMG_DIR.glob("*.jpg"):
                    f.unlink()
                st.rerun()

        # Initialize selection state
        if "del_selected" not in st.session_state:
            st.session_state["del_selected"] = set()

        # Select All / Clear: must set each checkbox key BEFORE they render
        if select_all:
            for p in catalog["products"]:
                st.session_state[f"chk_{p['sku']}"] = True
            st.session_state["del_selected"] = {p["sku"] for p in catalog["products"]}
        if clear_sel:
            for p in catalog["products"]:
                st.session_state[f"chk_{p['sku']}"] = False
            st.session_state["del_selected"] = set()

        # Product list with checkboxes + images
        for product in catalog["products"]:
            sku = product["sku"]
            cols = st.columns([0.3, 0.5, 2, 1.5, 1])
            with cols[0]:
                is_checked = st.checkbox("", key=f"chk_{sku}", label_visibility="collapsed")
                if is_checked:
                    st.session_state["del_selected"].add(sku)
                else:
                    st.session_state["del_selected"].discard(sku)
            with cols[1]:
                img_file = product.get("image_path", product.get("image", ""))
                if img_file:
                    img_path = REF_IMG_DIR / img_file
                    if img_path.is_file():
                        st.image(str(img_path), width=60)
                    else:
                        st.caption("🖼️")
                else:
                    st.caption("🖼️")
            with cols[2]:
                emb_icon = "✅" if product.get("embedding") else "❌"
                st.markdown(f"**{product['name']}** {emb_icon}")
            with cols[3]:
                st.caption(f"{sku} · {product.get('category', 'Other')}")
            with cols[4]:
                st.caption(f"₹{product.get('price', 0):.0f}")

        # Delete selected button
        selected = st.session_state.get("del_selected", set())
        if selected:
            st.markdown("---")
            if st.button(f"🗑️ Delete {len(selected)} Selected Product(s)", type="primary"):
                for sku in selected:
                    prod = next((p for p in catalog["products"] if p["sku"] == sku), None)
                    if prod:
                        img_file = prod.get("image_path", "")
                        if img_file:
                            img_p = REF_IMG_DIR / img_file
                            if img_p.is_file():
                                img_p.unlink()
                    delete_product(sku)
                st.session_state["del_selected"] = set()
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ── TAB 2: PLANOGRAM CREATOR ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📋 Planogram Creator — Build Shelf Layouts</div>', unsafe_allow_html=True)
    st.caption("Create planograms by auto-detecting from a shelf image OR manually defining product positions.")

    catalog = load_catalog()
    n_products = len([p for p in catalog["products"] if p.get("embedding")])

    if n_products < 2:
        st.warning(f"⚠️ Register at least 2 products in the **Product Scanner** tab first. Currently: {n_products} products.")
        plano_mode = None
    else:
        st.success(f"✅ {n_products} products in database — ready to create planograms!", icon="✅")

        # If an edit was triggered, default to Manual Editor
        plano_modes = ["📸 Auto-Detect from Shelf Image", "✏️ Manual Planogram Editor"]
        default_mode_idx = 1 if "edit_planogram_data" in st.session_state else 0

        # Mode selection
        plano_mode = st.radio(
            "Creation Mode",
            plano_modes,
            index=default_mode_idx,
            horizontal=True, key="plano_mode"
        )

    # ── MANUAL PLANOGRAM EDITOR ──────────────────────────────────
    if plano_mode == "✏️ Manual Planogram Editor":
        # Check if editing existing planogram
        editing = "edit_planogram_data" in st.session_state
        edit_data = st.session_state.get("edit_planogram_data", {})
        edit_name = st.session_state.get("edit_planogram_name", "")

        if editing:
            st.markdown(f"##### ✏️ Editing Planogram: **{edit_name}**")
            st.caption("Modify products and quantities, then click Update to save changes.")
            if st.button("❌ Cancel Edit", key="cancel_edit"):
                st.session_state.pop("edit_planogram_name", None)
                st.session_state.pop("edit_planogram_data", None)
                st.rerun()
        else:
            st.markdown("##### ✏️ Manual Planogram Builder")
            st.caption("Select products and quantities for each shelf. 100% accurate — you define exactly what goes where.")

        # Pre-fill name and shelves from edit data
        default_name = edit_name if editing else "Manual_Shelf_1"
        default_shelves = edit_data.get("n_shelves", 2) if editing else 2

        manual_name = st.text_input("Planogram Name", value=default_name, key="manual_plano_name")
        n_shelves = st.number_input("Number of Shelves", min_value=1, max_value=10, value=int(default_shelves), key="manual_n_shelves")

        product_names = [p["name"] for p in catalog["products"] if p.get("embedding")]
        product_lookup = {p["name"]: p for p in catalog["products"] if p.get("embedding")}

        # Build default selections from edit data
        edit_shelves = edit_data.get("shelves", []) if editing else []

        manual_shelves = []
        for shelf_idx in range(int(n_shelves)):
            st.markdown(f"---\n**Shelf {shelf_idx + 1}**")

            # Get default products and quantities for this shelf from edit data
            default_products = []
            default_quantities = {}
            if shelf_idx < len(edit_shelves):
                shelf_data = edit_shelves[shelf_idx]
                from collections import Counter
                name_counts = Counter(p["name"] for p in shelf_data.get("products", []))
                default_products = [n for n in name_counts.keys() if n in product_names]
                default_quantities = dict(name_counts)

            shelf_cols = st.columns([3, 1])
            with shelf_cols[0]:
                selected_products = st.multiselect(
                    f"Products on Shelf {shelf_idx + 1}",
                    product_names,
                    default=default_products,
                    key=f"manual_shelf_{shelf_idx}_products"
                )
            with shelf_cols[1]:
                quantities = {}
                for prod_name in selected_products:
                    default_qty = default_quantities.get(prod_name, 1)
                    qty = st.number_input(
                        f"Qty: {prod_name[:15]}", min_value=1, max_value=20, value=int(default_qty),
                        key=f"manual_qty_{shelf_idx}_{prod_name}"
                    )
                    quantities[prod_name] = qty

            shelf_products = []
            pos = 0
            for prod_name in selected_products:
                prod = product_lookup[prod_name]
                for _ in range(quantities.get(prod_name, 1)):
                    shelf_products.append({
                        "position": pos,
                        "sku": prod["sku"],
                        "name": prod["name"],
                        "confidence": 1.0,
                        "bbox": [0, 0, 0, 0],
                    })
                    pos += 1
            manual_shelves.append({
                "level": shelf_idx + 1,
                "product_count": len(shelf_products),
                "products": shelf_products,
            })

            if shelf_products:
                from collections import Counter
                counts = Counter(p["name"] for p in shelf_products)
                summary = ", ".join(f"{n} ×{c}" for n, c in counts.items())
                st.info(f"📦 Shelf {shelf_idx+1}: {summary}")

        st.markdown("---")
        total_manual = sum(s["product_count"] for s in manual_shelves)
        if total_manual > 0:
            btn_label = f"✅ Update Planogram" if editing else "✅ Save Manual Planogram"
            if st.button(btn_label, type="primary", width='stretch'):
                # If editing with a new name, delete the old one
                if editing and manual_name != edit_name:
                    delete_planogram(edit_name)
                    old_ref = PLANOGRAM_DIR / f"{edit_name}_reference.jpg"
                    old_ref.unlink(missing_ok=True)

                planogram_data = {
                    "name": manual_name,
                    "created_at": datetime.now().isoformat(),
                    "n_shelves": int(n_shelves),
                    "total_products": total_manual,
                    "shelves": manual_shelves,
                }
                save_planogram(manual_name, planogram_data)

                # Clear edit state
                st.session_state.pop("edit_planogram_name", None)
                st.session_state.pop("edit_planogram_data", None)

                action = "Updated" if editing else "Saved"
                st.success(f"✅ {action} planogram **{manual_name}** with {int(n_shelves)} shelves and {total_manual} products!")
                st.balloons()
        else:
            st.info("Add products to at least one shelf to save.")

    # ── AUTO-DETECT FROM IMAGE ───────────────────────────────────
    elif plano_mode == "📸 Auto-Detect from Shelf Image":
        st.success(f"✅ {n_products} products in database — ready to create planograms!", icon="✅")

        # Shelf name and image
        plano_cols = st.columns([1, 2])
        with plano_cols[0]:
            shelf_name = st.text_input("Shelf Name", value="Shelf_1", placeholder="e.g., Aisle_1_Shelf_A")
            shelf_image_source = st.radio(
                "Image Source",
                ["📱 Phone Camera", "💻 Laptop Camera", "📁 Upload"],
                horizontal=True, key="plano_source"
            )

        with plano_cols[1]:
            shelf_image = None
            if shelf_image_source == "📱 Phone Camera":
                st.info("Point your phone at the arranged shelf, then click capture 👇")
                if st.button("📸 Capture Shelf from Phone", type="primary", key="phone_cap_plano"):
                    shelf_image = capture_from_phone(phone_cam_url, phone_rotation)
                    if shelf_image:
                        st.session_state["plano_phone_img"] = shelf_image
                # Persist across reruns
                if "plano_phone_img" in st.session_state:
                    shelf_image = st.session_state["plano_phone_img"]
                    st.image(shelf_image, caption="Captured shelf", width='stretch')

            elif shelf_image_source == "💻 Laptop Camera":
                cam_img = st.camera_input("Capture your arranged shelf", key="plano_cam")
                if cam_img:
                    shelf_image = Image.open(cam_img).convert("RGB")
            else:
                uploaded = st.file_uploader("Upload shelf photo", type=["jpg", "jpeg", "png"], key="plano_upload")
                if uploaded:
                    shelf_image = Image.open(uploaded).convert("RGB")

        if shelf_image:
            with st.spinner("🔍 Analyzing shelf layout..."):
                yolo = load_yolo()
                dinov2 = load_dinov2()

                if yolo and dinov2:
                    # Step 1: Detect products
                    detections, results = run_detection(yolo, shelf_image, conf=0.25)

                    # Step 2: Detect shelf levels
                    boundaries, n_shelves = detect_shelf_levels(detections, shelf_image.height)

                    # Step 3: Assign to shelves
                    shelf_assignments = assign_to_shelves(detections, boundaries)

                    # Step 4: Identify each product
                    faiss_index, index_products = build_faiss_index(catalog)
                    product_labels = []

                    for det in detections:
                        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                        crop = shelf_image.crop((x1, y1, x2, y2))
                        emb = get_embedding(dinov2, crop)
                        match, score = search_product_with_size(
                            faiss_index, index_products, emb,
                            query_bbox=(x1, y1, x2, y2),
                            threshold=0.3
                        )
                        if match:
                            det["product_name"] = match["name"]
                            det["product_sku"] = match["sku"]
                            det["match_score"] = score
                            product_labels.append(match["name"])
                        else:
                            det["product_name"] = f"Unknown_{len(product_labels)+1}"
                            det["product_sku"] = "UNKNOWN"
                            det["match_score"] = score
                            product_labels.append(f"Unknown")

                    # Show annotated image
                    annotated = draw_annotated_image(shelf_image, detections, product_labels)
                    st.image(annotated, caption=f"Detected {len(detections)} products on {n_shelves} shelves", width='stretch')

                    # Show detected layout
                    st.markdown("##### 📊 Auto-Detected Layout")
                    planogram_data = {
                        "name": shelf_name,
                        "created_at": datetime.now().isoformat(),
                        "n_shelves": n_shelves,
                        "total_products": len(detections),
                        "shelves": [],
                    }

                    for shelf_id in sorted(shelf_assignments.keys()):
                        shelf_dets = shelf_assignments[shelf_id]
                        products_on_shelf = []
                        product_summary = []

                        for pos, det in enumerate(shelf_dets):
                            products_on_shelf.append({
                                "position": pos,
                                "sku": det.get("product_sku", "UNKNOWN"),
                                "name": det.get("product_name", "Unknown"),
                                "confidence": round(det.get("match_score", 0), 3),
                                "bbox": det["bbox"],
                            })
                            product_summary.append(det.get("product_name", "Unknown"))

                        planogram_data["shelves"].append({
                            "level": shelf_id,
                            "product_count": len(shelf_dets),
                            "products": products_on_shelf,
                        })

                        # Count products by name
                        from collections import Counter
                        counts = Counter(product_summary)
                        summary = ", ".join(f"{name} ×{count}" for name, count in counts.items())
                        st.markdown(f'<div class="alert-info"><strong>Shelf {shelf_id}:</strong> {summary} ({len(shelf_dets)} products)</div>', unsafe_allow_html=True)

                    # Confirm button
                    st.markdown("")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Confirm as Planogram", type="primary", width='stretch'):
                            save_planogram(shelf_name, planogram_data)
                            # Also save the reference image
                            shelf_image.save(str(PLANOGRAM_DIR / f"{shelf_name}_reference.jpg"), "JPEG", quality=90)
                            st.success(f"✅ Planogram **{shelf_name}** saved with {n_shelves} shelves and {len(detections)} products!")
                            st.balloons()
                    with c2:
                        if st.button("🔄 Re-scan", width='stretch'):
                            st.rerun()

    # Show existing planograms
    planograms = load_planograms()
    if planograms:
        st.markdown("---")
        st.markdown("##### 📋 Saved Planograms")
        for name, data in planograms.items():
            with st.expander(f"📄 {name} — {data.get('n_shelves', '?')} shelves, {data.get('total_products', '?')} products"):
                for shelf in data.get("shelves", []):
                    products = [p["name"] for p in shelf.get("products", [])]
                    from collections import Counter
                    counts = Counter(products)
                    summary = ", ".join(f"{n} ×{c}" for n, c in counts.items())
                    st.markdown(f"**Shelf {shelf['level']}:** {summary}")

                # Reference image
                ref_img = PLANOGRAM_DIR / f"{name}_reference.jpg"
                if ref_img.exists():
                    st.image(str(ref_img), caption=f"Reference: {name}", width='stretch')

                btn_cols = st.columns([1, 1, 3])
                with btn_cols[0]:
                    if st.button(f"✏️ Edit {name}", key=f"edit_{name}"):
                        # Load planogram data into session state for editing
                        st.session_state["edit_planogram_name"] = name
                        st.session_state["edit_planogram_data"] = data
                        st.rerun()
                with btn_cols[1]:
                    if st.button(f"🗑️ Delete {name}", key=f"del_{name}"):
                        delete_planogram(name)
                        ref_img.unlink(missing_ok=True)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ── TAB 3: LIVE MONITORING ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🎥 Live Shelf Monitoring — Real-Time Compliance</div>', unsafe_allow_html=True)
    st.caption("Start monitoring to continuously watch your shelf and auto-detect planogram violations.")

    planograms = load_planograms()
    catalog = load_catalog()
    n_products = len([p for p in catalog["products"] if p.get("embedding")])

    if not planograms:
        st.warning("⚠️ Create a planogram first in the **Planogram Creator** tab.")
    elif n_products < 2:
        st.warning("⚠️ Register products first in the **Product Scanner** tab.")
    else:
        # Controls
        ctrl_cols = st.columns([2, 1, 1])
        with ctrl_cols[0]:
            selected_planogram = st.selectbox("Select Planogram to Check Against", list(planograms.keys()))
        with ctrl_cols[1]:
            conf_threshold = st.slider("Detection Confidence", 0.15, 0.9, 0.25, 0.05)
        with ctrl_cols[2]:
            ntfy_topic = st.text_input("Notification Topic", value="shelfmind-alerts", help="Install ntfy app → subscribe to this topic")
            st.session_state["ntfy_topic"] = ntfy_topic

        scan_interval = st.slider("Scan Interval (seconds)", 3, 30, 5, 1, help="How often to capture and analyze a new frame")

        # Camera source selection
        cam_source = st.radio(
            "Camera Source",
            ["💻 Laptop Webcam", "📱 Phone Camera (IP Webcam)"],
            horizontal=True,
            help="Use IP Webcam app on Android for better quality"
        )
        ip_cam_url = ""
        if cam_source == "📱 Phone Camera (IP Webcam)":
            ip_cam_url = st.text_input(
                "IP Webcam URL",
                value="http://192.168.1.5:8080/video",
                help="Install 'IP Webcam' app on Android → Start Server → Use the URL shown"
            )
            st.markdown("""<div class="alert-info">
                <strong>📱 Setup IP Webcam:</strong><br>
                1. Install <strong>IP Webcam</strong> app on Android from Play Store<br>
                2. Open app → scroll down → tap <strong>Start Server</strong><br>
                3. Note the URL shown (e.g., http://192.168.1.5:8080)<br>
                4. Add <strong>/video</strong> at the end and paste above<br>
                5. Make sure phone & laptop are on <strong>same WiFi</strong>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Real-time monitoring controls
        btn_cols = st.columns([1, 1, 2])
        with btn_cols[0]:
            start_monitoring = st.button("▶️ Start Live Monitoring", type="primary", width='stretch')
        with btn_cols[1]:
            stop_monitoring = st.button("⏹️ Stop Monitoring", width='stretch')

        if stop_monitoring:
            st.session_state["monitoring_active"] = False

        if start_monitoring:
            st.session_state["monitoring_active"] = True

        # Initialize session state
        if "monitoring_active" not in st.session_state:
            st.session_state["monitoring_active"] = False
        if "last_alert_time" not in st.session_state:
            st.session_state["last_alert_time"] = 0

        # Placeholders for real-time updates
        status_indicator = st.empty()
        metric_row = st.empty()
        frame_display = st.empty()
        compliance_report = st.empty()
        alert_log = st.empty()
        chart_display = st.empty()

        if st.session_state.get("monitoring_active", False):
            import cv2

            planogram = planograms[selected_planogram]
            yolo = load_yolo()
            dinov2 = load_dinov2()
            faiss_index, index_products = build_faiss_index(catalog)

            if not yolo or not dinov2 or faiss_index is None:
                st.error("❌ Models not loaded. Please check YOLO and DINOv2.")
                st.session_state["monitoring_active"] = False
            else:
                # Determine camera source
                use_phone = cam_source == "📱 Phone Camera (IP Webcam)" and ip_cam_url
                cam_label = f"Phone Camera ({ip_cam_url})" if use_phone else "Laptop Webcam"

                status_indicator.markdown(
                    f'<div class="alert-ok"><strong>🟢 LIVE MONITORING ACTIVE</strong> — {cam_label} is watching the shelf. Any violation will trigger an alert.</div>',
                    unsafe_allow_html=True
                )

                # For phone: use HTTP shot grab (more reliable than video stream)
                # For laptop: use OpenCV VideoCapture
                cap = None
                if not use_phone:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error(f"❌ Cannot access laptop webcam.")
                        st.session_state["monitoring_active"] = False

                scan_count = 0
                retry_count = 0
                max_retries = 5
                # Multi-frame voting buffer: stores per-shelf SKU counts from last N frames
                VOTE_BUFFER_SIZE = 3
                detection_history = []  # List of per-frame shelf SKU counts

                if st.session_state.get("monitoring_active", False):
                    try:
                        while st.session_state.get("monitoring_active", False):
                            monitor_image = None

                            if use_phone:
                                # Grab single frame via HTTP (reliable, no stream drops)
                                monitor_image = capture_from_phone(ip_cam_url, phone_rotation if 'phone_rotation' in dir() else 90)
                                if not monitor_image:
                                    retry_count += 1
                                    if retry_count > max_retries:
                                        status_indicator.markdown(
                                            '<div class="alert-critical"><strong>❌ Phone camera unreachable after 5 retries.</strong> Check IP Webcam app.</div>',
                                            unsafe_allow_html=True
                                        )
                                        break
                                    status_indicator.markdown(
                                        f'<div class="alert-warning"><strong>⚠️ Reconnecting to phone... (attempt {retry_count}/{max_retries})</strong></div>',
                                        unsafe_allow_html=True
                                    )
                                    time.sleep(3)
                                    continue
                                else:
                                    retry_count = 0  # Reset on success
                            else:
                                ret, frame = cap.read()
                                if not ret:
                                    status_indicator.markdown(
                                        '<div class="alert-critical"><strong>❌ Camera feed lost.</strong> Reconnecting...</div>',
                                        unsafe_allow_html=True
                                    )
                                    time.sleep(2)
                                    continue
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                monitor_image = Image.fromarray(frame_rgb)

                            scan_count += 1
                            current_time = datetime.now()

                            # Resize for performance
                            max_dim = 640
                            if monitor_image.width > max_dim:
                                ratio = max_dim / monitor_image.width
                                monitor_image = monitor_image.resize(
                                    (max_dim, int(monitor_image.height * ratio)),
                                    Image.LANCZOS,
                                )

                            # ── DETECTION ──────────────────────────────────────
                            detections, _ = run_detection(yolo, monitor_image, conf=conf_threshold)
                            boundaries, n_shelves = detect_shelf_levels(detections, monitor_image.height)
                            shelf_assignments = assign_to_shelves(detections, boundaries)

                            # ── SKU IDENTIFICATION (DINOv2 + Size-Ratio Fusion) ──
                            # Collect all expected products from planogram for size comparison
                            all_expected_products = []
                            for ps in planogram.get("shelves", []):
                                all_expected_products.extend(ps.get("products", []))

                            for det in detections:
                                x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                                crop = monitor_image.crop((max(0, x1), max(0, y1), x2, y2))
                                emb = get_embedding(dinov2, crop)
                                match, score = search_product_with_size(
                                    faiss_index, index_products, emb,
                                    query_bbox=(x1, y1, x2, y2),
                                    expected_products_on_shelf=all_expected_products,
                                    threshold=0.3,
                                    size_weight=0.15
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

                            # ── MULTI-FRAME VOTING ────────────────────────────
                            # Collect this frame's per-shelf SKU counts
                            from collections import Counter
                            frame_shelf_counts = {}
                            for shelf_id, shelf_dets in shelf_assignments.items():
                                sku_counts = Counter(
                                    d.get("product_sku", "UNKNOWN") for d in shelf_dets
                                    if d.get("product_sku") != "UNKNOWN"
                                )
                                frame_shelf_counts[shelf_id] = dict(sku_counts)
                            detection_history.append(frame_shelf_counts)
                            if len(detection_history) > VOTE_BUFFER_SIZE:
                                detection_history.pop(0)

                            # Voted counts: for each shelf+SKU, use median count across frames
                            voted_shelf_counts = {}
                            if len(detection_history) >= 2:
                                all_shelf_ids = set()
                                for fsc in detection_history:
                                    all_shelf_ids.update(fsc.keys())
                                for sid in all_shelf_ids:
                                    all_skus = set()
                                    for fsc in detection_history:
                                        all_skus.update(fsc.get(sid, {}).keys())
                                    voted_shelf_counts[sid] = {}
                                    for sku in all_skus:
                                        counts_across_frames = [
                                            fsc.get(sid, {}).get(sku, 0)
                                            for fsc in detection_history
                                        ]
                                        # Use median (robust to outliers)
                                        voted_shelf_counts[sid][sku] = int(
                                            sorted(counts_across_frames)[len(counts_across_frames) // 2]
                                        )

                            # ── PLANOGRAM COMPARISON (uses voted counts when available) ─
                            plan_shelves = planogram.get("shelves", [])
                            all_alerts = []
                            shelf_compliance = {}
                            total_expected = 0
                            total_matched = 0

                            for plan_shelf in plan_shelves:
                                shelf_id = plan_shelf["level"]
                                expected_products = plan_shelf.get("products", [])
                                detected_on_shelf = shelf_assignments.get(shelf_id, [])

                                from collections import Counter
                                expected_counts = Counter(p["sku"] for p in expected_products if p["sku"] != "UNKNOWN")
                                # Use voted counts if available (multi-frame), else raw
                                if voted_shelf_counts and shelf_id in voted_shelf_counts:
                                    detected_counts = Counter(voted_shelf_counts[shelf_id])
                                else:
                                    detected_counts = Counter(d.get("product_sku", "UNKNOWN") for d in detected_on_shelf if d.get("product_sku") != "UNKNOWN")

                                issues = []
                                shelf_expected = len(expected_products)
                                shelf_matched = 0
                                revenue_at_risk = 0

                                for sku, expected_count in expected_counts.items():
                                    detected_count = detected_counts.get(sku, 0)
                                    prod_name = next((p["name"] for p in expected_products if p["sku"] == sku), sku)
                                    prod_price = next((d.get("product_price", 0) for d in detected_on_shelf if d.get("product_sku") == sku), 0)

                                    if detected_count == 0:
                                        issues.append(f"🔴 **{prod_name}** — MISSING (expected {expected_count}, found 0)")
                                        revenue_at_risk += prod_price * expected_count
                                        all_alerts.append({
                                            "type": "STOCKOUT", "shelf": shelf_id,
                                            "product": prod_name, "sku": sku,
                                            "expected": expected_count, "found": 0,
                                            "revenue": prod_price * expected_count,
                                            "priority": "CRITICAL",
                                        })
                                    elif detected_count < expected_count:
                                        missing = expected_count - detected_count
                                        issues.append(f"⚠️ **{prod_name}** — LOW STOCK ({detected_count}/{expected_count} facings)")
                                        revenue_at_risk += prod_price * missing
                                        all_alerts.append({
                                            "type": "LOW_STOCK", "shelf": shelf_id,
                                            "product": prod_name, "sku": sku,
                                            "expected": expected_count, "found": detected_count,
                                            "revenue": prod_price * missing,
                                            "priority": "HIGH",
                                        })
                                        shelf_matched += detected_count
                                    else:
                                        shelf_matched += expected_count

                                for sku, count in detected_counts.items():
                                    if sku not in expected_counts:
                                        prod_name = next((d.get("product_name", sku) for d in detected_on_shelf if d.get("product_sku") == sku), sku)
                                        issues.append(f"🚫 **{prod_name}** — UNAUTHORIZED (not in planogram)")
                                        all_alerts.append({
                                            "type": "UNAUTHORIZED", "shelf": shelf_id,
                                            "product": prod_name, "sku": sku,
                                            "priority": "MEDIUM",
                                        })

                                # ── POSITION/ORDER CHECK ──────────────────────────
                                # Compare left-to-right order of detected vs expected
                                expected_order = [p["sku"] for p in expected_products if p["sku"] != "UNKNOWN"]
                                detected_sorted = sorted(
                                    [d for d in detected_on_shelf if d.get("product_sku", "UNKNOWN") != "UNKNOWN"],
                                    key=lambda d: d["bbox"][0]  # Sort by x1 (left to right)
                                )
                                detected_order = [d.get("product_sku") for d in detected_sorted]

                                # Check if order matches
                                if expected_order and detected_order:
                                    min_len = min(len(expected_order), len(detected_order))
                                    for pos_idx in range(min_len):
                                        if expected_order[pos_idx] != detected_order[pos_idx]:
                                            # Find what's at this position
                                            expected_name = next(
                                                (p["name"] for p in expected_products if p["sku"] == expected_order[pos_idx]),
                                                expected_order[pos_idx]
                                            )
                                            detected_name = next(
                                                (d.get("product_name", "?") for d in detected_sorted if d.get("product_sku") == detected_order[pos_idx]),
                                                detected_order[pos_idx]
                                            )
                                            issues.append(
                                                f"🔄 **{detected_name}** — MISPLACED (position {pos_idx+1}: expected **{expected_name}**)"
                                            )
                                            all_alerts.append({
                                                "type": "MISPLACED", "shelf": shelf_id,
                                                "product": detected_name,
                                                "expected_at": expected_name,
                                                "position": pos_idx + 1,
                                                "priority": "HIGH",
                                            })
                                            # Reduce compliance for misplacement
                                            if shelf_matched > 0:
                                                shelf_matched -= 0.5  # Half penalty for wrong order

                                if not issues:
                                    issues.append("All products in correct position")

                                comp_pct = (shelf_matched / shelf_expected * 100) if shelf_expected > 0 else 100
                                shelf_compliance[shelf_id] = {
                                    "compliance": comp_pct, "expected": shelf_expected,
                                    "detected": len(detected_on_shelf), "matched": shelf_matched,
                                    "issues": issues, "revenue_at_risk": revenue_at_risk,
                                }
                                total_expected += shelf_expected
                                total_matched += shelf_matched

                            overall_compliance = (total_matched / total_expected * 100) if total_expected > 0 else 100
                            total_revenue_risk = sum(s["revenue_at_risk"] for s in shelf_compliance.values())

                            # ── UPDATE UI (all placeholders) ──────────────────
                            with metric_row.container():
                                m1, m2, m3, m4, m5 = st.columns(5)
                                with m1:
                                    color = "#00d4aa" if overall_compliance >= 80 else "#ffaa00" if overall_compliance >= 50 else "#ff4343"
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{overall_compliance:.1f}%</div>
                                        <div class="metric-label">Compliance</div>
                                    </div>""", unsafe_allow_html=True)
                                with m2:
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-value">{len(detections)}</div>
                                        <div class="metric-label">Detected</div>
                                    </div>""", unsafe_allow_html=True)
                                with m3:
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-value">{total_expected}</div>
                                        <div class="metric-label">Expected</div>
                                    </div>""", unsafe_allow_html=True)
                                with m4:
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-value" style="background: #ff4343; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">₹{total_revenue_risk:.0f}</div>
                                        <div class="metric-label">Revenue at Risk</div>
                                    </div>""", unsafe_allow_html=True)
                                with m5:
                                    st.markdown(f"""<div class="metric-card">
                                        <div class="metric-value">#{scan_count}</div>
                                        <div class="metric-label">Scan Count</div>
                                    </div>""", unsafe_allow_html=True)

                            # Annotated frame
                            annotated = draw_annotated_image(monitor_image, detections)
                            vote_status = f"🗳️ Voted ({len(detection_history)}/{VOTE_BUFFER_SIZE} frames)" if len(detection_history) >= 2 else "⏳ Warming up..."
                            frame_display.image(annotated, caption=f"🔴 LIVE — Scan #{scan_count} at {current_time.strftime('%H:%M:%S')} | {len(detections)} products | {vote_status}", width='stretch')

                            # Compliance report
                            with compliance_report.container():
                                st.markdown("##### 📋 Real-Time Compliance Status")
                                for shelf_id in sorted(shelf_compliance.keys()):
                                    data = shelf_compliance[shelf_id]
                                    alert_html = format_compliance_alert(
                                        f"Shelf {shelf_id}", data["issues"], data["compliance"],
                                    )
                                    st.markdown(alert_html, unsafe_allow_html=True)

                            # ── MOBILE PUSH (throttle: max 1 per 30 seconds) ──
                            critical_alerts = [a for a in all_alerts if a["priority"] in ("CRITICAL", "HIGH")]
                            time_since_last = time.time() - st.session_state.get("last_alert_time", 0)

                            if critical_alerts and ntfy_topic and time_since_last > 30:
                                alert_msg = f"🔴 ShelfMind Live Alert — {selected_planogram}\n"
                                alert_msg += f"Compliance: {overall_compliance:.0f}% | Scan #{scan_count}\n"
                                alert_msg += f"Time: {current_time.strftime('%H:%M:%S')}\n\n"
                                for a in critical_alerts[:5]:
                                    type_emoji = {"STOCKOUT": "X", "LOW_STOCK": "!", "MISPLACED": "->", "UNAUTHORIZED": "??"}.get(a["type"], "!")
                                    alert_msg += f"[{type_emoji}] {a['product']}: {a['type']} on Shelf {a['shelf']}\n"
                                if total_revenue_risk > 0:
                                    alert_msg += f"\nRevenue at risk: ₹{total_revenue_risk:.0f}/hr"

                                sent = send_mobile_alert(
                                    f"🔴 Shelf Alert — {overall_compliance:.0f}% Compliance",
                                    alert_msg,
                                    "urgent" if overall_compliance < 50 else "high",
                                )
                                if sent:
                                    st.session_state["last_alert_time"] = time.time()
                                    with alert_log.container():
                                        st.markdown(f"""<div class="alert-critical">
                                            <strong>📱 Alert Sent at {current_time.strftime('%H:%M:%S')}</strong><br>
                                            {len(critical_alerts)} violation(s) detected → Push notification sent to <strong>ntfy.sh/{ntfy_topic}</strong>
                                        </div>""", unsafe_allow_html=True)

                            # Save compliance log to SQLite database
                            comp_log_id = log_compliance(
                                planogram_name=selected_planogram,
                                compliance=round(overall_compliance, 1),
                                detected=len(detections),
                                expected=total_expected,
                                revenue_risk=round(total_revenue_risk, 2),
                                alert_count=len(all_alerts),
                                scan_number=scan_count,
                            )

                            # Log individual alerts to database
                            for a in all_alerts:
                                log_alert(
                                    compliance_log_id=comp_log_id,
                                    alert_type=a.get("type", "UNKNOWN"),
                                    shelf_id=a.get("shelf", 0),
                                    product_name=a.get("product", ""),
                                    product_sku=a.get("sku", ""),
                                    priority=a.get("priority", "MEDIUM"),
                                    expected_count=a.get("expected"),
                                    found_count=a.get("found"),
                                    revenue=a.get("revenue", 0),
                                    position_info=a.get("expected_at", ""),
                                    notified=bool(critical_alerts),
                                )

                            # Wait before next scan
                            time.sleep(scan_interval)

                    except Exception as e:
                        st.error(f"Monitoring error: {e}")
                    finally:
                        if cap is not None:
                            cap.release()
                        st.session_state["monitoring_active"] = False
                        status_indicator.markdown(
                            '<div class="alert-warning"><strong>⏹️ Monitoring stopped.</strong> Click ▶️ Start to resume.</div>',
                            unsafe_allow_html=True
                        )
        else:
            # Not monitoring — show instructions
            st.markdown("""<div class="alert-info">
                <strong>📋 How Real-Time Monitoring Works:</strong><br>
                1. Select the planogram to compare against<br>
                2. Set the notification topic (install <strong>ntfy</strong> app on your phone)<br>
                3. Click <strong>▶️ Start Live Monitoring</strong><br>
                4. The system will automatically:<br>
                &nbsp;&nbsp;&nbsp;• Capture frames from your camera<br>
                &nbsp;&nbsp;&nbsp;• Detect & identify all products<br>
                &nbsp;&nbsp;&nbsp;• Compare against the planogram<br>
                &nbsp;&nbsp;&nbsp;• Show violations in real-time<br>
                &nbsp;&nbsp;&nbsp;• Send push notifications to your phone 📱<br>
                5. Try removing or misplacing a product — watch the alert fire!
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# ── TAB 4: ANALYTICS DASHBOARD ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📊 Analytics Dashboard — Shelf Intelligence at a Glance</div>', unsafe_allow_html=True)
    st.caption("Real-time overview of shelf health, compliance trends, and revenue impact across your store.")

    # Load compliance logs from database
    logs = get_compliance_logs_as_list()

    catalog = load_catalog()
    planograms = load_planograms()

    # ── Top Metrics ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    n_scans = len(logs)
    avg_compliance = np.mean([l["overall_compliance"] for l in logs]) if logs else 0
    total_rev_risk = sum(l.get("revenue_at_risk", 0) for l in logs)
    total_alerts = sum(l.get("alerts", 0) for l in logs)

    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(catalog.get('products', []))}</div>
            <div class="metric-label">Products in DB</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(planograms)}</div>
            <div class="metric-label">Active Planograms</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        color = "#00d4aa" if avg_compliance >= 80 else "#ffaa00" if avg_compliance >= 50 else "#ff4343"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{avg_compliance:.1f}%</div>
            <div class="metric-label">Avg Compliance</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n_scans}</div>
            <div class="metric-label">Total Scans</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="background: #ff4343; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">₹{total_rev_risk:.0f}</div>
            <div class="metric-label">Total Revenue at Risk</div>
        </div>""", unsafe_allow_html=True)

    if logs:
        st.markdown("---")
        chart_cols = st.columns(2)

        # Compliance trend
        with chart_cols[0]:
            log_df = pd.DataFrame(logs)
            log_df["time"] = pd.to_datetime(log_df["timestamp"])
            fig_trend = px.line(log_df, x="time", y="overall_compliance",
                              title="📈 Compliance Trend Over Time",
                              markers=True, line_shape="spline")
            fig_trend.update_traces(line_color="#00d4aa", line_width=3)
            fig_trend.add_hline(y=80, line_dash="dash", line_color="#ffaa00",
                               annotation_text="Target: 80%")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=350,
                yaxis_title="Compliance %",
                xaxis_title="Time",
            )
            st.plotly_chart(fig_trend, width='stretch')

        # Alert distribution
        with chart_cols[1]:
            fig_alerts = px.bar(log_df, x="time", y="alerts",
                               title="⚠️ Alert Frequency",
                               color="overall_compliance",
                               color_continuous_scale=["#ff4343", "#ffaa00", "#00d4aa"])
            fig_alerts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=350,
            )
            st.plotly_chart(fig_alerts, width='stretch')

        # Shelf-level heatmap
        st.markdown("##### 🗺️ Shelf Health Heatmap")
        if logs[-1].get("shelf_data"):
            latest = logs[-1]["shelf_data"]
            shelf_names = [f"Shelf {k}" for k in sorted(latest.keys())]
            compliances = [latest[k]["compliance"] for k in sorted(latest.keys())]

            fig_heatmap = go.Figure(data=go.Bar(
                x=shelf_names, y=compliances,
                marker=dict(
                    color=compliances,
                    colorscale=[[0, "#ff4343"], [0.5, "#ffaa00"], [1.0, "#00d4aa"]],
                    cmin=0, cmax=100,
                ),
                text=[f"{c:.0f}%" for c in compliances],
                textposition="auto",
            ))
            fig_heatmap.update_layout(
                title="Shelf-Level Compliance Scores",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300,
                yaxis_title="Compliance %",
            )
            st.plotly_chart(fig_heatmap, width='stretch')

        # Recent alerts table
        st.markdown("##### 📋 Recent Compliance Scans")
        display_df = log_df[["timestamp", "planogram", "overall_compliance", "total_detected", "total_expected", "alerts", "revenue_at_risk"]].copy()
        display_df.columns = ["Timestamp", "Planogram", "Compliance %", "Detected", "Expected", "Alerts", "Revenue at Risk (₹)"]
        st.dataframe(display_df.tail(20).sort_index(ascending=False), width='stretch', hide_index=True)

    else:
        st.info("📊 No compliance data yet. Run a compliance check in the **Live Monitor** tab to see analytics here.")

        # Demo data for visual appeal
        st.markdown("##### 📊 Demo Analytics Preview")
        demo_cols = st.columns(2)
        with demo_cols[0]:
            dates = pd.date_range(end=datetime.now(), periods=14, freq="D")
            demo_compliance = 65 + np.cumsum(np.random.randn(14) * 2)
            demo_compliance = np.clip(demo_compliance, 40, 100)
            fig = px.line(x=dates, y=demo_compliance, title="📈 Compliance Trend (Demo)",
                         markers=True, line_shape="spline")
            fig.update_traces(line_color="#00d4aa", line_width=3)
            fig.add_hline(y=80, line_dash="dash", line_color="#ffaa00")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="white", height=300)
            st.plotly_chart(fig, width='stretch')

        with demo_cols[1]:
            hours = list(range(8, 22))
            aisles = ["Aisle 1", "Aisle 2", "Aisle 3", "Aisle 4"]
            heatmap_data = np.random.uniform(60, 100, size=(len(aisles), len(hours)))
            heatmap_data[1, 3:6] = [30, 25, 40]  # Stockout period
            fig = px.imshow(heatmap_data, x=[f"{h}:00" for h in hours], y=aisles,
                           title="🗺️ Stockout Heatmap (Demo)",
                           color_continuous_scale=["#ff4343", "#ffaa00", "#00d4aa"],
                           zmin=0, zmax=100, aspect="auto")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="white", height=300)
            st.plotly_chart(fig, width='stretch')



# ══════════════════════════════════════════════════════════════════════════
# ── TAB 5: TRAINING RESULTS ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📓 Model Training Results</div>', unsafe_allow_html=True)
    st.caption("Performance metrics from YOLO, RF-DETR, DINOv2, and LightGBM training.")

    # ── Detection Model Comparison ──
    st.markdown("##### 🏆 Detection Model Comparison — YOLO26s vs RF-DETR")

    comp_cols = st.columns(2)
    with comp_cols[0]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.2rem; background: linear-gradient(135deg, #00d4aa, #00a98f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">YOLO26s v2 (Fine-tuned)</div>
            <div class="metric-label">NMS-Free · Lightning.ai A100 · 60 epochs · 1280px</div>
            <br>
            <table style="width: 100%; color: #8892b0; font-size: 0.85rem;">
                <tr><td>mAP@50</td><td style="text-align: right; color: #00d4aa; font-weight: bold;">0.917 🏆</td></tr>
                <tr><td>mAP@50-95</td><td style="text-align: right; color: #00d4aa;">0.583</td></tr>
                <tr><td>Precision</td><td style="text-align: right;">0.912</td></tr>
                <tr><td>Recall</td><td style="text-align: right;">0.872</td></tr>
                <tr><td>F1 Score</td><td style="text-align: right;">0.891</td></tr>
                <tr><td>Parameters</td><td style="text-align: right;">9.9M</td></tr>
                <tr><td>Model Size</td><td style="text-align: right;">76.7 MB</td></tr>
                <tr><td>Inference</td><td style="text-align: right; color: #00d4aa;">~8 ms/img</td></tr>
                <tr><td>Training Time</td><td style="text-align: right;">4.6 hrs</td></tr>
                <tr><td>GPU</td><td style="text-align: right;">A100-SXM4 (80GB)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with comp_cols[1]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.2rem; background: linear-gradient(135deg, #7c3aed, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RF-DETR-Base (Fine-tuned)</div>
            <div class="metric-label">Transformer · Lightning.ai L4 · 30 epochs</div>
            <br>
            <table style="width: 100%; color: #8892b0; font-size: 0.85rem;">
                <tr><td>mAP@50</td><td style="text-align: right; color: #a78bfa; font-weight: bold;">0.887</td></tr>
                <tr><td>mAP@50-95</td><td style="text-align: right; color: #a78bfa;">0.547</td></tr>
                <tr><td>Precision</td><td style="text-align: right;">0.911</td></tr>
                <tr><td>Recall</td><td style="text-align: right;">0.847</td></tr>
                <tr><td>F1 Score</td><td style="text-align: right;">0.878</td></tr>
                <tr><td>Parameters</td><td style="text-align: right;">~29M</td></tr>
                <tr><td>Model Size</td><td style="text-align: right;">~130 MB</td></tr>
                <tr><td>Inference</td><td style="text-align: right;">~15 ms/img</td></tr>
                <tr><td>Training Time</td><td style="text-align: right;">~4 hrs</td></tr>
                <tr><td>GPU</td><td style="text-align: right;">L4 (24GB)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # ── Key Insights ──
    st.markdown("---")
    st.markdown("##### 📊 Key Insights")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.markdown("""<div class="alert-ok">
            <strong>🏆 YOLO26s v2 — Best mAP</strong><br>
            • mAP@50 = 0.917 (beats YOLOv10s at 0.906)<br>
            • 1280px resolution → 4× more detail<br>
            • State-of-the-art on SKU-110K
        </div>""", unsafe_allow_html=True)
    with insight_cols[1]:
        st.markdown("""<div class="alert-info">
            <strong>🎯 RF-DETR: DINOv2 Backbone</strong><br>
            • Transformer attention for global context<br>
            • DINOv2 features for richer embeddings<br>
            • Training RF-DETR-Large in progress
        </div>""", unsafe_allow_html=True)
    with insight_cols[2]:
        st.markdown("""<div class="alert-info">
            <strong>📈 v1 → v2 Improvement</strong><br>
            • mAP@50: 0.895 → 0.917 (+2.2%)<br>
            • Recall: 0.848 → 0.872 (+2.4%)<br>
            • AdamW + Cosine LR + copy-paste aug
        </div>""", unsafe_allow_html=True)

    # ── Training Config Comparison Table ──
    st.markdown("---")
    st.markdown("##### ⚙️ Training Configuration Comparison")

    import pandas as pd
    config_data = {
        "Config": ["Dataset", "Images (Train/Val)", "Annotations", "Epochs", "Batch Size",
                    "Effective Batch", "Optimizer", "Learning Rate", "Resolution", "Platform"],
        "YOLO26s v2": ["SKU-110K", "8,219 / 588", "1.2M bboxes", "60 (patience=30)",
                     "16", "64 (NBS)", "AdamW + Cosine LR", "0.001 → 0.00001",
                     "1280×1280", "Lightning.ai A100-80GB"],
        "RF-DETR-Base": ["SKU-110K", "8,219 / 588", "1.2M bboxes", "30", "2",
                          "16 (grad_accum=8)", "AdamW", "1e-4", "Multi-scale", "Lightning.ai L4"],
    }
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, hide_index=True, use_container_width=True)

    # ── Existing Training Visualizations ──
    st.markdown("---")
    st.markdown("##### 📈 Training Visualizations")

    if VIZ_DIR.exists():
        viz_files = sorted(VIZ_DIR.glob("*.png"))
        if viz_files:
            # Group into rows of 2
            for i in range(0, len(viz_files), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(viz_files):
                        with col:
                            st.image(str(viz_files[i+j]), caption=viz_files[i+j].stem.replace("_", " ").title(), width='stretch')
        else:
            st.info("No training visualizations found in `models/training_visualizations/`.")
    else:
        st.info("Run training on Kaggle first to generate visualization files.")

    # Model info cards
    st.markdown("---")
    st.markdown("##### 🏗️ Model Architecture")
    arch_cols = st.columns(4)
    with arch_cols[0]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">YOLO26s</div>
            <div class="metric-label">Object Detection</div>
            <br>
            <small style="color: #8892b0;">
            • NMS-free architecture<br>
            • Trained on SKU-110K dataset<br>
            • 43% faster on CPU vs YOLO11<br>
            • STAL: small-target-aware<br>
            • 9.9M params · 22.5 GFLOPs
            </small>
        </div>""", unsafe_allow_html=True)
    with arch_cols[1]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">RF-DETR-Base</div>
            <div class="metric-label">Object Detection</div>
            <br>
            <small style="color: #8892b0;">
            • Transformer attention (ICLR 2026)<br>
            • DINOv2 ViT backbone<br>
            • Trained on SKU-110K dataset<br>
            • Higher precision for dense shelves<br>
            • ~29M params
            </small>
        </div>""", unsafe_allow_html=True)
    with arch_cols[2]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">DINOv2 ViT-S/14</div>
            <div class="metric-label">SKU Recognition</div>
            <br>
            <small style="color: #8892b0;">
            • Self-supervised learning<br>
            • 768-dim embeddings (fine-tuned)<br>
            • 384-dim embeddings (pretrained)<br>
            • FAISS cosine similarity search<br>
            • 15-view robust augmentation
            </small>
        </div>""", unsafe_allow_html=True)
    with arch_cols[3]:
        st.markdown("""<div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">LightGBM</div>
            <div class="metric-label">Demand Forecasting</div>
            <br>
            <small style="color: #8892b0;">
            • Trained on Walmart M5 data<br>
            • 15 features (temporal, price, SNAP)<br>
            • SHAP explainability<br>
            • Auto-replenishment alerts
            </small>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""<div class="footer">
    ShelfMind AI — Smart Retail Shelf Intelligence — Built for Hackathon 2026<br>
    <small>YOLO26s + RF-DETR + DINOv2 + FAISS + LightGBM + SHAP | Computer Vision-Driven Inventory Monitoring</small>
</div>""", unsafe_allow_html=True)
