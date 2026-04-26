# ShelfMind AI — Web Application

> **Smart Retail Shelf Intelligence** powered by YOLO26s + DINOv2 + FAISS

## 🏗️ Project Structure

```
shelfmind-web/
├── backend/                    # FastAPI Backend (Deployed on HuggingFace Spaces)
│   ├── api_server.py          # Main API server — all ML endpoints
│   ├── db.py                  # SQLite database layer
│   ├── Dockerfile             # Docker config for HF Spaces
│   ├── README.md              # HF Spaces metadata
│   └── models/                # ML model weights (Git LFS)
│       ├── yolo_shelf_best.pt          # YOLO26s detection (20 MB)
│       ├── dinov2_shelf_finetuned.pth  # DINOv2 embeddings (330 MB)
│       └── dinov2_projector.pth        # Projection head (24 MB)
│
├── frontend/                   # Next.js 15 Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx     # Root layout + sidebar
│   │   │   ├── globals.css    # Obsidian Prism design system
│   │   │   ├── page.tsx       # Dashboard
│   │   │   ├── scanner/
│   │   │   │   └── page.tsx   # Product Scanner (single + bulk)
│   │   │   ├── planogram/
│   │   │   │   └── page.tsx   # Planogram Creator (auto + manual)
│   │   │   ├── monitor/
│   │   │   │   └── page.tsx   # Live Monitor (compliance)
│   │   │   ├── analytics/
│   │   │   │   └── page.tsx   # Analytics Dashboard
│   │   │   └── training/
│   │   │       └── page.tsx   # Training Results
│   │   ├── components/
│   │   │   └── Sidebar.tsx    # Navigation sidebar
│   │   └── lib/
│   │       └── api.ts         # API client config
│   ├── package.json
│   ├── next.config.ts
│   └── tsconfig.json
│
└── README.md                   # This file
```

## 🚀 Quick Start

### Backend (Already Deployed)
Live at: **https://kush5699-shelfmind-ai.hf.space**

### Frontend (Local Development)
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:3000**

## 🔑 Features

| Page | Features |
|------|----------|
| **Dashboard** | Health monitoring, model status, neural pipeline, activity timeline |
| **Product Scanner** | Single/bulk scan, webcam/upload, barcode + OCR + rembg auto-crop, voice registration |
| **Planogram Creator** | Auto-detect shelves, manual editor, save/deploy planograms |
| **Live Monitor** | Real-time compliance, per-shelf status, incident log, push alerts |
| **Analytics** | KPI cards, compliance trends, alert composition, top offenders |
| **Training Results** | Model metrics, architecture flow, training curves |

## 🎨 Design System: Obsidian Prism

- **Theme:** Dark editorial glassmorphism
- **Primary:** `#00d4aa` (Teal)
- **Secondary:** `#00b4d8` (Blue)  
- **Accent:** `#7b68ee` (Purple)
- **Font:** Inter (Variable)
- **Cards:** 24px radius, glass blur, no-border separation
