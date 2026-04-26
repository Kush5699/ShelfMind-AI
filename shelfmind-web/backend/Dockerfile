FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libzbar0 git curl && \
    rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces requires non-root user
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Python dependencies (install as root first)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart aiofiles \
    ultralytics faiss-cpu timm joblib \
    numpy pandas scikit-learn Pillow opencv-python-headless \
    easyocr "rembg[cpu]" requests tqdm pyzbar

# Pre-download rembg U2-Net model
RUN python -c "from rembg import new_session; new_session('u2net')"

# Pre-download EasyOCR English model
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Pre-download DINOv2 base from torch.hub
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)"

# Copy our custom trained model files
COPY --chown=user models/yolo_shelf_best.pt /app/models/yolo_shelf_best.pt
COPY --chown=user models/dinov2_shelf_finetuned.pth /app/models/dinov2_shelf_finetuned.pth
COPY --chown=user models/dinov2_projector.pth /app/models/dinov2_projector.pth

# Copy backend code
COPY --chown=user api_server.py /app/api_server.py
COPY --chown=user db.py /app/db.py

# Create data directories with user permissions
RUN mkdir -p /app/data/store_catalog/reference_images /app/data/store_planograms /app/data/compliance_logs && \
    chown -R user:user /app/data

# Switch to non-root user
USER user

EXPOSE 7860

CMD ["python", "api_server.py"]
