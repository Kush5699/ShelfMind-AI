<p align="center">
  <h1 align="center">ShelfMind AI</h1>
  <p align="center"><strong>AI-Powered Smart Retail Shelf Intelligence System</strong></p>
  <p align="center">
    <em>Real-time product detection, visual SKU matching, demand forecasting & planogram compliance</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLO-v11s-00d4aa" />
  <img src="https://img.shields.io/badge/DINOv2-ViT--S-4ecdc4" />
  <img src="https://img.shields.io/badge/LightGBM-4.0-ffe66d" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff6b6b" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
</p>

---

## Problem Statement

Retail stores lose **$1.1 trillion annually** to stockouts and shelf mismanagement ([IHL Group](https://www.ihlservices.com/)). Store associates manually checking thousands of shelf positions is slow, error-prone, and expensive. ShelfMind AI automates this entire workflow using computer vision and machine learning.

### Target Persona
**Store Manager / Operations Associate** -- needs real-time visibility into shelf status, automated restock alerts, and planogram compliance without manual shelf walks.

### Must-Have Features Addressed
- **Product Detection** on retail shelves (dense, overlapping products)
- **SKU Identification** via visual similarity matching
- **Demand Forecasting** for proactive restocking
- **Planogram Compliance** to detect shelf layout violations
- **Interactive Dashboard** for real-time monitoring

---

## System Architecture

```
                         +------------------+
                         |  Camera / Upload |
                         +--------+---------+
                                  |
                    +-------------v--------------+
                    |   YOLO11s Product Detector  |  <-- Trained on SKU-110K
                    |   (9.4M params, 21.5 GFLOPs) |     1.7M+ bounding boxes
                    +---+-------------------+-----+
                        |                   |
              +---------v--------+  +-------v--------+
              | Product Counting |  | Crop Extraction |
              | & Shelf Mapping  |  | (per detection) |
              +--------+---------+  +-------+--------+
                       |                    |
                       |          +---------v----------+
                       |          | DINOv2 ViT-S/14    |  <-- 384-dim embeddings
                       |          | Visual Encoder      |
                       |          +---------+----------+
                       |                    |
                       |          +---------v----------+
                       |          | FAISS Index Search  |  <-- 2,984 SKU vectors
                       |          | (Cosine Similarity) |     <1ms per query
                       |          +---------+----------+
                       |                    |
              +--------v--------------------v--------+
              |    Planogram Compliance Engine        |
              |  Expected vs Detected layout check   |
              +--------+----------------------------++
                       |                             |
              +--------v--------+          +---------v---------+
              | LightGBM Demand |          |   Restock Alert   |
              | Forecasting     |          |   Generator       |
              | (M5 Walmart)    |          +-------------------+
              +--------+--------+
                       |
              +--------v----------------------------+
              |   Streamlit Dashboard (Real-time)   |
              |   - Shelf Analysis Tab              |
              |   - Demand Forecast Tab             |
              |   - Planogram Compliance Tab        |
              |   - Training Results Tab            |
              +-------------------------------------+
```

---

## Technical Design

### Pipeline Architecture (Modular & Extensible)

| Layer | Component | Technology | Why This Choice |
|-------|-----------|-----------|-----------------|
| **Detection** | Product Localization | YOLO11s (Ultralytics) | SOTA real-time detector, handles dense shelf packing via mosaic augmentation |
| **Embedding** | Visual Feature Extraction | DINOv2 ViT-S/14 (Meta) | Self-supervised ViT with strong zero-shot transfer, no fine-tuning needed |
| **Retrieval** | SKU Matching | FAISS (Facebook) | Sub-millisecond cosine similarity search over 2,984+ product vectors |
| **Forecasting** | Demand Prediction | LightGBM | Handles high-cardinality features, fast inference, interpretable via SHAP |
| **Frontend** | Interactive Dashboard | Streamlit + Plotly | Rapid prototyping, real-time widgets, zero frontend code needed |
| **Data Pipeline** | Feature Engineering | Pandas + NumPy | Lag features, rolling stats, calendar encoding, multi-source fusion |

### Modular Extensibility

ShelfMind AI uses **decoupled components**:

- **Swap detector:** Replace YOLO11s with any Ultralytics model (YOLOv8, RT-DETR) via 1-line config change
- **Swap embeddings:** Replace DINOv2 with CLIP/SigLIP by changing the embedding model loader
- **Add data sources:** Weather API, price feeds, or IoT sensor data plug into the forecasting pipeline via feature columns
- **API-ready:** Each model exposes independent prediction functions, ready for FastAPI/MCP wrapping

### Integration Logic

```
User Upload --> [Streamlit UI] --> YOLO inference (GPU/CPU)
                                       |
                                       +--> crop images --> DINOv2 --> FAISS search
                                       |
                                       +--> count + positions --> Planogram check
                                       |
                              [LightGBM] --> forecast --> restock alert
```

All layers communicate via **in-memory Python objects** (NumPy arrays, PIL Images) -- zero serialization overhead. Dashboard uses `@st.cache_resource` for model persistence across requests.

---

## Performance Metrics

### Product Detection (YOLO11s on SKU-110K)

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | **90.65%** | Low false positive rate |
| **Recall** | **82.71%** | Strong detection coverage |
| **mAP@50** | **86.76%** | Primary detection metric |
| **mAP@50-95** | **54.05%** | Strict IoU evaluation |
| **Inference** | **7.3ms/image** | Real-time capable (137 FPS) |
| Training | 30 epochs, 2.2 hrs on T4 | 8,219 train images, 1.2M bboxes |

### SKU Matching (DINOv2 + FAISS)

| Metric | Value |
|--------|-------|
| **Index Size** | 2,984 product crop embeddings |
| **Embedding Dim** | 384 (ViT-S/14) |
| **Query Latency** | < 1ms per search |
| **Similarity** | Cosine (normalized L2) |

### Demand Forecasting (LightGBM on M5 Walmart)

| Metric | Value | Notes |
|--------|-------|-------|
| **MAE** | **6.20 units** | Average prediction error |
| **RMSE** | **9.66 units** | Penalizes large errors |
| **Early Stopped** | Epoch 172/500 | No overfitting |
| **Top Features** | Rolling mean (7d), Lag (14d), Price | Interpretable & business-relevant |

### System Efficiency

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection latency | < 200ms | **7.3ms** (27x faster) |
| SKU search latency | < 100ms | **< 1ms** (100x faster) |
| Model total size | < 100MB | **24.8 MB** |
| Dashboard cold start | < 10s | **~5s** |

---

## Innovation & Advanced Techniques

### 1. Multi-Model Fusion Pipeline
Unlike single-model approaches, ShelfMind chains **3 specialized models** (YOLO -> DINOv2 -> LightGBM), each best-in-class for its task. Detection crops feed directly into visual embeddings, which feed into SKU-aware forecasting.

### 2. Self-Supervised Visual Search (No SKU Labels Needed)
DINOv2 creates meaningful product embeddings **without any labeled SKU data**. New products are indexed automatically -- just detect, embed, and add to FAISS. No retraining required.

### 3. Dense Scene Detection
SKU-110K is a **densely packed** retail dataset (avg 147 products/image). Standard object detectors fail on such scenes. Our YOLO11s with mosaic augmentation + DFL loss handles extreme product density.

### 4. Temporal Feature Engineering
The forecasting model uses **multi-source fusion**:
- **Sales signals:** 7/14/28-day lags, rolling mean/std
- **Calendar features:** Day of week, month, holidays, SNAP food assistance days
- **Price features:** Sell price (captures promotions/elasticity)
- **Categorical encoding:** Store, department, category hierarchies

### 5. Explainability (Feature Importance)
LightGBM provides built-in feature importance (split count), making predictions interpretable. The dashboard shows **why** a demand spike is predicted (e.g., upcoming holiday + SNAP day + price drop).

---

## Project Structure

```
ShelfMind-AI/
|
+-- app/
|   +-- dashboard.py              # Streamlit interactive dashboard (780 lines)
|
+-- kaggle_training/
|   +-- shelfmind_train.py        # Complete Kaggle training pipeline
|                                  # YOLO + DINOv2 + LightGBM + 9 visualizations
|
+-- scripts/
|   +-- generate_weather_data.py   # Open-Meteo API weather fetcher
|   +-- generate_planogram_data.py # Synthetic planogram generator (10 stores)
|
+-- models/shelfmind_models/
|   +-- yolo_shelf_best.pt         # YOLO11s detector (19.2 MB)
|   +-- sku_faiss_index.bin        # FAISS product index (4.6 MB)
|   +-- sku_metadata.json          # Crop metadata (0.5 MB)
|   +-- lgbm_forecast_model.pkl    # LightGBM model (0.6 MB)
|
+-- data/                          # Datasets (not in repo)
|   +-- m5-forecasting-accuracy/   # Walmart M5 sales
|   +-- planograms/                # Store planogram JSONs
|   +-- weather/                   # Historical weather data
|
+-- README.md
+-- requirements.txt
+-- LICENSE
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Kush5699/ShelfMind-AI.git
cd ShelfMind-AI
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard loads all models automatically and provides 4 tabs:
- **Shelf Analysis:** Upload images, detect products, match SKUs
- **Demand Forecasting:** Interactive forecast simulator with restock alerts
- **Planogram Compliance:** Compare detected vs expected shelf layouts
- **Training Results:** View all training curves and visualizations

### 3. Train from Scratch (Kaggle)

```bash
# 1. Create Kaggle Notebook with GPU T4
# 2. Add "m5-forecasting-accuracy" dataset
# 3. Paste kaggle_training/shelfmind_train.py
# 4. Run all cells (~2.2 hours)
# 5. Download shelfmind_models.zip from Output tab
```

---

## Datasets

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) | 11,762 images, 1.7M bboxes | CVPR 2019 | YOLO product detection training |
| [M5 Walmart Sales](https://www.kaggle.com/c/m5-forecasting-accuracy) | 30,490 items x 1,913 days | Kaggle | LightGBM demand forecasting |
| [Supermarket Shelves](https://universe.roboflow.com/) | 45 images + annotations | Roboflow | Additional shelf validation |
| Weather Data | 3 states x 5 years | Open-Meteo API | Forecast feature enrichment |

---

## Resilience & Data Handling

- **Corrupt image handling:** YOLO auto-repairs truncated JPEGs during training (300+ restored in SKU-110K)
- **Missing data:** LightGBM natively handles NaN features; `dropna()` applied after lag computation
- **Disk constraints:** Streaming extraction (`curl | tar`) bypasses Kaggle's 19.5 GB limit
- **Model fallbacks:** Dashboard gracefully degrades if any model file is missing (shows warnings, other tabs still work)
- **Cache strategy:** `@st.cache_resource` ensures models are loaded once across all dashboard sessions

---

## Scalability & Production Readiness

### Current Scale
- **Detection:** 8,219 training images, 1.2M bounding boxes
- **SKU Index:** 2,984 product embeddings (expandable to 100K+ with FAISS IVF)
- **Forecasting:** 500 items x 365 days (expandable to 30K+ items)

### Path to Production
| Aspect | Current | Production Path |
|--------|---------|----------------|
| **Deployment** | Streamlit (local) | Docker + FastAPI + Nginx |
| **SKU Scale** | 2,984 products | FAISS IVF index (1M+ products) |
| **Inference** | Single GPU/CPU | TensorRT / ONNX optimization |
| **Data Pipeline** | Batch CSV | Apache Kafka + real-time streams |
| **Monitoring** | Dashboard visuals | Prometheus + Grafana alerts |
| **Multi-store** | 10 planograms | Database-backed store configs |

### ROI Potential
- **Stockout reduction:** 30-40% fewer missed sales from proactive restock alerts
- **Labor savings:** 2-3 hours/day per store from automated shelf auditing
- **Planogram compliance:** From 70% manual accuracy to 95%+ automated verification

---

## Validation & Testing

### Training Validation
- **YOLO:** Validated on 588 held-out val images (90,968 bboxes) from SKU-110K
- **LightGBM:** Time-based split (last 28 days as test), early stopping at epoch 172/500
- **Visualizations:** 9 auto-generated charts (training curves, detection samples, t-SNE, feature importance, actual vs predicted)

### Dashboard Testing
- All 4 tabs functional with trained model artifacts
- Graceful degradation when models are missing
- Tested on RTX 5050 (8GB VRAM) for local inference

---

## Tech Stack Summary

```
Computer Vision:  YOLO11s + DINOv2 ViT-S/14
Vector Search:    FAISS (Facebook AI Similarity Search)
Forecasting:      LightGBM + Pandas feature engineering
Dashboard:        Streamlit + Plotly
Training:         Kaggle GPU T4 (Tesla T4, 15.6 GB)
Languages:        Python 3.10+
```

---

## License

MIT License -- see [LICENSE](LICENSE)

---

<p align="center">
  <strong>ShelfMind AI</strong> -- Making every shelf smarter<br>
  <em>Built for Smart Retail Hackathon 2026</em>
</p>
