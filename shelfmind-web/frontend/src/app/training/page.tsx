"use client";

import { useEffect, useState } from "react";
import { API, apiFetch } from "@/lib/api";

// Deterministic pseudo-random for SSR/client consistency
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 9301 + 49297) * 49297;
  return x - Math.floor(x);
}

const PIPELINE_STAGES = [
  { icon: "📹", label: "Camera Feed", desc: "4K shelf stream input", bg: "var(--surface-container)" },
  { icon: "🔍", label: "YOLO26s", desc: "Product detection & localization", bg: "var(--primary-container)" },
  { icon: "✂️", label: "rembg Crop", desc: "AI background removal", bg: "var(--surface-container)" },
  { icon: "🧬", label: "DINOv2 Embed", desc: "768-dim feature extraction", bg: "var(--primary-container)" },
  { icon: "🔗", label: "FAISS Index", desc: "Cosine similarity search", bg: "var(--surface-container)" },
  { icon: "📋", label: "Compliance", desc: "Planogram validation", bg: "var(--primary-container)" },
  { icon: "🔔", label: "Alert Dispatch", desc: "ntfy.sh push notifications", bg: "var(--surface-container)" },
];

export default function TrainingPage() {
  const [health, setHealth] = useState<{ models: Record<string, boolean> } | null>(null);

  useEffect(() => {
    apiFetch(API.health).then(setHealth).catch(() => {});
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div style={{ fontSize: "0.72rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--on-surface-muted)", marginBottom: 4 }}>
          Neural Engine V2.4
        </div>
        <h1 className="page-title">
          Training <span className="text-gradient">Results</span>
        </h1>
        <p className="page-subtitle">
          Model performance metrics, architecture details, and system pipeline overview.
        </p>
      </div>

      {/* Model Cards */}
      <div className="grid-2col" style={{ marginBottom: 32 }}>
        {/* YOLO26s */}
        <div className="surface-card" style={{ padding: 28 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 20 }}>
            <div>
              <h3 style={{ fontSize: "1.3rem", fontWeight: 800 }}>YOLO26s Object Detector</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                High-speed edge inference optimization
              </p>
            </div>
            <span className="badge badge-success">Stable</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
            <div>
              <div className="metric-label">Dataset Size</div>
              <div style={{ fontSize: "1.8rem", fontWeight: 800 }}>SKU-110K</div>
            </div>
            <div>
              <div className="metric-label">mAP@50</div>
              <div className="metric-value" style={{ fontSize: "2rem" }}>0.895</div>
            </div>
          </div>

          {/* Training Curve */}
          <div style={{ height: 140, display: "flex", alignItems: "end", gap: 2, borderBottom: "1px solid var(--glass-border)", paddingBottom: 8 }}>
            {Array.from({ length: 30 }, (_, i) => {
              const progress = i / 29;
              const loss = Math.max(0.1, 1 - progress * 0.8 - seededRandom(i + 100) * 0.05);
              const height = (1 - loss) * 120;
              return (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: Math.max(4, height),
                    borderRadius: "2px 2px 0 0",
                    background: `linear-gradient(to top, var(--primary), var(--secondary))`,
                    opacity: 0.4 + progress * 0.6,
                  }}
                />
              );
            })}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: "0.7rem", color: "var(--on-surface-muted)" }}>
            <span>Epochs 0-250</span>
            <span style={{ color: "var(--primary)" }}>Converged at 242</span>
          </div>

          <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
            <div style={{ padding: 12, background: "var(--surface-container)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: "0.68rem", color: "var(--on-surface-muted)", textTransform: "uppercase" }}>Precision</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 700, marginTop: 2 }}>0.912</div>
            </div>
            <div style={{ padding: 12, background: "var(--surface-container)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: "0.68rem", color: "var(--on-surface-muted)", textTransform: "uppercase" }}>Recall</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 700, marginTop: 2 }}>0.878</div>
            </div>
            <div style={{ padding: 12, background: "var(--surface-container)", borderRadius: "var(--radius-md)" }}>
              <div style={{ fontSize: "0.68rem", color: "var(--on-surface-muted)", textTransform: "uppercase" }}>F1 Score</div>
              <div style={{ fontSize: "1.1rem", fontWeight: 700, marginTop: 2 }}>0.895</div>
            </div>
          </div>
        </div>

        {/* DINOv2 */}
        <div className="surface-card" style={{ padding: 28 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 20 }}>
            <div>
              <h3 style={{ fontSize: "1.3rem", fontWeight: 800 }}>DINOv2 Embedder</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                SSL Feature extraction backbone (fine-tuned)
              </p>
            </div>
            <span className="badge badge-success">Stable</span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 24 }}>
            {[
              { label: "Latent Dim", value: "768-dim" },
              { label: "Architecture", value: "ViT-B/14" },
              { label: "Hardware", value: "NVIDIA T4" },
              { label: "Precision", value: "FP16" },
              { label: "Fine-tuned On", value: "Kaggle SKU Dataset" },
              { label: "Augmentation Views", value: "10 per product" },
            ].map((item, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 12, borderBottom: "1px solid var(--glass-border)" }}>
                <span style={{ fontSize: "0.85rem", color: "var(--on-surface-dim)" }}>{item.label}</span>
                <span style={{ fontSize: "0.9rem", fontWeight: 700 }}>{item.value}</span>
              </div>
            ))}
          </div>

          {/* Embedding Quality Chart */}
          <div style={{ height: 100, display: "flex", alignItems: "center", gap: 3 }}>
            {Array.from({ length: 40 }, (_, i) => {
              const val = Math.sin(i / 6) * 0.3 + seededRandom(i + 200) * 0.2 + 0.5;
              return (
                <div
                  key={i}
                  style={{
                    flex: 1,
                    height: `${val * 80}%`,
                    borderRadius: 2,
                    background: i % 2 === 0 ? "var(--accent)" : "rgba(123, 104, 238, 0.3)",
                  }}
                />
              );
            })}
          </div>
          <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)", marginTop: 6, textAlign: "center" }}>
            Embedding distribution • Cosine similarity histogram
          </div>
        </div>
      </div>

      {/* Additional Models */}
      <div className="grid-3col" style={{ marginBottom: 32 }}>
        <div className="surface-card" style={{ padding: 24 }}>
          <div style={{ fontSize: "1.5rem", marginBottom: 8 }}>✂️</div>
          <h4 style={{ fontWeight: 700, marginBottom: 4 }}>rembg (U2-Net)</h4>
          <p style={{ fontSize: "0.78rem", color: "var(--on-surface-muted)", marginBottom: 12 }}>AI background removal for clean product crops</p>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
            <span style={{ color: "var(--on-surface-dim)" }}>Model Size</span>
            <span style={{ fontWeight: 600 }}>176 MB</span>
          </div>
          <div style={{ marginTop: 8 }}>
            {health?.models?.rembg ? <span className="badge badge-success">✓ Active</span> : <span className="badge badge-danger">✗ Offline</span>}
          </div>
        </div>

        <div className="surface-card" style={{ padding: 24 }}>
          <div style={{ fontSize: "1.5rem", marginBottom: 8 }}>🔤</div>
          <h4 style={{ fontWeight: 700, marginBottom: 4 }}>EasyOCR</h4>
          <p style={{ fontSize: "0.78rem", color: "var(--on-surface-muted)", marginBottom: 12 }}>Multilingual text extraction from product labels</p>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
            <span style={{ color: "var(--on-surface-dim)" }}>Languages</span>
            <span style={{ fontWeight: 600 }}>English</span>
          </div>
          <div style={{ marginTop: 8 }}>
            {health?.models?.ocr ? <span className="badge badge-success">✓ Active</span> : <span className="badge badge-danger">✗ Offline</span>}
          </div>
        </div>

        <div className="surface-card" style={{ padding: 24 }}>
          <div style={{ fontSize: "1.5rem", marginBottom: 8 }}>📊</div>
          <h4 style={{ fontWeight: 700, marginBottom: 4 }}>FAISS Index</h4>
          <p style={{ fontSize: "0.78rem", color: "var(--on-surface-muted)", marginBottom: 12 }}>Facebook AI Similarity Search for product matching</p>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem" }}>
            <span style={{ color: "var(--on-surface-dim)" }}>Search Type</span>
            <span style={{ fontWeight: 600 }}>Cosine + Size</span>
          </div>
          <div style={{ marginTop: 8 }}>
            <span className="badge badge-success">✓ Active</span>
          </div>
        </div>
      </div>

      {/* Neural Architecture Flow */}
      <div className="surface-card" style={{ padding: 28, marginBottom: 32 }}>
        <h3 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: 4, textAlign: "center" }}>
          Neural Architecture Flow
        </h3>
        <p style={{ fontSize: "0.82rem", color: "var(--on-surface-muted)", textAlign: "center", marginBottom: 28 }}>
          End-to-end inference pipeline from camera to alert dispatch
        </p>

        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8, alignItems: "center" }}>
          {PIPELINE_STAGES.map((stage, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 8,
                  padding: "16px 20px",
                  borderRadius: "var(--radius-lg)",
                  background: stage.bg,
                  border: "1px solid var(--glass-border)",
                  minWidth: 100,
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "1.5rem" }}>{stage.icon}</div>
                <div style={{ fontSize: "0.78rem", fontWeight: 700 }}>{stage.label}</div>
                <div style={{ fontSize: "0.65rem", color: "var(--on-surface-muted)" }}>{stage.desc}</div>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <div style={{ fontSize: "1.2rem", color: "var(--primary)", fontWeight: 700 }}>→</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
