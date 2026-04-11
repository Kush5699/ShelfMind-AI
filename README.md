# ShelfMind AI - Smart Retail Shelf Intelligence

> **AI-powered retail shelf monitoring system** that combines computer vision, visual search, and demand forecasting to optimize retail operations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

ShelfMind AI is an end-to-end intelligent retail shelf management system that:

1. **Detects products** on retail shelves using YOLO11 (trained on SKU-110K, 1.7M+ bounding boxes)
2. **Matches SKUs** via DINOv2 visual embeddings + FAISS similarity search
3. **Forecasts demand** using LightGBM on Walmart M5 sales data (30K+ products, 1900+ days)
4. **Checks planogram compliance** by comparing detected layouts against store planograms

## Architecture

```
Camera Feed --> YOLO11 Detection --> DINOv2 Embedding --> FAISS SKU Match
                    |                                          |
                    v                                          v
            Product Count                              SKU Identification
                    |                                          |
                    +---> Planogram Compliance Check <----------+
                    |
                    v
            LightGBM Forecast --> Restock Alerts
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Detection | YOLO11s + Ultralytics | Product localization on shelves |
| SKU Matching | DINOv2 ViT-S + FAISS | Visual product identification |
| Forecasting | LightGBM | Demand prediction (M5 Walmart) |
| Backend | FastAPI | REST API for inference |
| Dashboard | Streamlit | Interactive monitoring UI |
| Training | Kaggle T4 GPU | Model training pipeline |

## Model Performance

### Product Detection (YOLO11s on SKU-110K)
| Metric | Value |
|--------|-------|
| Precision | **90.7%** |
| Recall | **82.7%** |
| mAP@50 | **86.8%** |
| mAP@50-95 | **54.1%** |
| Training | 30 epochs, 2.2 hours on T4 |
| Dataset | 8,219 train + 588 val + 2,936 test images |

### SKU Matching (DINOv2 + FAISS)
| Metric | Value |
|--------|-------|
| Index Size | 2,984 product embeddings |
| Embedding Dim | 384 |
| Search | Cosine similarity (< 1ms per query) |

### Demand Forecasting (LightGBM on M5)
| Metric | Value |
|--------|-------|
| MAE | 6.2 units |
| RMSE | 9.7 units |
| Top Features | Rolling mean (7d), Sales lag (14d) |
| Dataset | 500 top items x 365 days |

## Project Structure

```
Smart Retail Shelf Intelligence/
|-- kaggle_training/
|   |-- shelfmind_train.py      # Complete Kaggle training pipeline
|
|-- scripts/
|   |-- generate_weather_data.py # Weather data fetcher (Open-Meteo API)
|   |-- generate_planogram_data.py # Planogram JSON generator
|
|-- models/                      # Trained model artifacts (from Kaggle)
|   |-- yolo_shelf_best.pt       # YOLO11s product detector
|   |-- sku_faiss_index.bin      # DINOv2 FAISS index
|   |-- sku_metadata.json        # SKU crop metadata
|   |-- lgbm_forecast_model.pkl  # LightGBM forecasting model
|
|-- data/                        # Datasets (not in repo, see below)
|   |-- m5-forecasting-accuracy/ # Walmart M5 sales data
|   |-- shelf_images/            # Supermarket shelf images
|   |-- planograms/              # Store planogram JSONs
|   |-- weather/                 # Historical weather data
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/ShelfMind-AI.git
cd ShelfMind-AI
pip install -r requirements.txt
```

### 2. Download Models
Download trained models from [Kaggle Notebook Output](YOUR_KAGGLE_LINK) or [GitHub Releases](YOUR_RELEASE_LINK):
```bash
# Place in models/ directory
models/
  yolo_shelf_best.pt
  sku_faiss_index.bin
  sku_metadata.json
  lgbm_forecast_model.pkl
```

### 3. Train from Scratch (Optional)
```bash
# Run on Kaggle with GPU T4
# 1. Create new notebook, add M5 dataset
# 2. Paste kaggle_training/shelfmind_train.py
# 3. Run all cells (~2 hours)
```

## Datasets Used

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) | 11,762 images, 1.7M bboxes | CVPR 2019 | YOLO training |
| [M5 Walmart Sales](https://www.kaggle.com/c/m5-forecasting-accuracy) | 30,490 items x 1,913 days | Kaggle | Demand forecasting |
| Supermarket Shelves | 45 images + annotations | Roboflow | Additional shelf samples |

## Training Pipeline

The complete training runs on **Kaggle T4 GPU** in a single notebook:

1. **YOLO11s** - Fine-tuned on SKU-110K (auto-downloaded, 11.4 GB streamed)
2. **DINOv2 ViT-S** - Extracts 384-dim embeddings from detected product crops
3. **FAISS Index** - Built from 2,984 product embeddings for real-time SKU search
4. **LightGBM** - Trained on M5 sales with lag/rolling/calendar features

## License

MIT License

## Team

Built for the Smart Retail Hackathon

---

*ShelfMind AI - Making every shelf smarter*
