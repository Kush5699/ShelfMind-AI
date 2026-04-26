"use client";

import { useEffect, useState } from "react";
import { API, apiFetch } from "@/lib/api";

interface HealthData {
  status: string;
  device: string;
  gpu: string;
  models: { yolo: boolean; dinov2: boolean; ocr: boolean; rembg: boolean };
  products_registered: number;
  timestamp: string;
}

const PIPELINE_STEPS = [
  { icon: "📹", label: "Input", desc: "4K RAW Stream" },
  { icon: "🔍", label: "YOLO26s Detect", desc: "Object Localization" },
  { icon: "✂️", label: "Crop", desc: "Instance Normalization" },
  { icon: "🧬", label: "DINOv2 Embed", desc: "Feature Extraction" },
  { icon: "🔗", label: "FAISS Match", desc: "Index Search" },
  { icon: "✅", label: "Output", desc: "Telemetry Validated" },
];

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [summary, setSummary] = useState<Record<string, any>>({});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [recentLogs, setRecentLogs] = useState<any[]>([]);

  useEffect(() => {
    Promise.all([
      apiFetch(API.health).catch(() => null),
      apiFetch(API.analyticsSummary).catch(() => ({})),
      apiFetch(API.complianceHistory).catch(() => ({ logs: [] })),
    ]).then(([h, s, logsRes]) => {
      setHealth(h);
      setSummary(s || {});
      setRecentLogs((logsRes?.logs || []).slice(0, 5));
      setLoading(false);
    });
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <span className="badge badge-success">
            <span className="pulse-dot pulse-dot-green" />
            LIVE MONITORING ACTIVE
          </span>
        </div>
        <h1 className="page-title">
          Welcome back, <span className="text-gradient">Manager</span>
        </h1>
        <p className="page-subtitle">
          System operational. Neural engines are processing live shelf telemetry across all zones.
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid-metrics" style={{ marginBottom: 32 }}>
        <div className="metric-card">
          <div className="metric-label">Scans Recorded</div>
          <div className="metric-value">{summary?.total_scans ?? recentLogs.length ?? 0}</div>
          <div className="metric-subtitle" style={{ color: "var(--primary)" }}>Total compliance checks</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Products</div>
          <div className="metric-value">{loading ? "—" : health?.products_registered ?? 0}</div>
          <div className="metric-subtitle">Active SKU Registry</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg Compliance</div>
          <div className="metric-value">
            {summary?.avg_compliance ? `${summary.avg_compliance.toFixed(1)}%` : recentLogs.length > 0 ? `${(recentLogs.reduce((s: number, l: { compliance: number }) => s + l.compliance, 0) / recentLogs.length).toFixed(1)}%` : "—"}
          </div>
          <div className="metric-subtitle">Across all planograms</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">System</div>
          <div className="metric-value metric-value-sm">{health?.gpu ?? "CPU"}</div>
          <div className="metric-subtitle">{health?.device ?? "loading"}</div>
        </div>
      </div>

      {/* Model Status + Pipeline */}
      <div className="grid-60-40" style={{ marginBottom: 32 }}>
        {/* Neural Pipeline */}
        <div className="surface-card" style={{ padding: 28 }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
            ⚡ Neural Pipeline
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {PIPELINE_STEPS.map((step, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 16, position: "relative" }}>
                {/* Connector line */}
                {i < PIPELINE_STEPS.length - 1 && (
                  <div
                    style={{
                      position: "absolute",
                      left: 19,
                      top: 44,
                      width: 2,
                      height: 32,
                      background: "linear-gradient(to bottom, var(--primary), transparent)",
                      opacity: 0.3,
                    }}
                  />
                )}
                {/* Icon */}
                <div
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: "var(--radius-md)",
                    background: "var(--surface-container)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1.1rem",
                    flexShrink: 0,
                  }}
                >
                  {step.icon}
                </div>
                {/* Text */}
                <div style={{ padding: "12px 0" }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600 }}>{step.label}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--on-surface-muted)" }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Status Cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="surface-card" style={{ padding: 24 }}>
            <div className="metric-label">Detector</div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 4 }}>YOLO26s</div>
            <div className="metric-subtitle">FP16 Precision</div>
            <div style={{ marginTop: 12 }}>
              {health?.models?.yolo ? (
                <span className="badge badge-success">✓ Loaded</span>
              ) : loading ? (
                <span className="badge badge-info">Loading...</span>
              ) : (
                <span className="badge badge-danger">✗ Offline</span>
              )}
            </div>
          </div>
          <div className="surface-card" style={{ padding: 24 }}>
            <div className="metric-label">Embedder</div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 4 }}>DINOv2</div>
            <div className="metric-subtitle">ViT-L/14 Arch</div>
            <div style={{ marginTop: 12 }}>
              {health?.models?.dinov2 ? (
                <span className="badge badge-success">✓ Loaded</span>
              ) : loading ? (
                <span className="badge badge-info">Loading...</span>
              ) : (
                <span className="badge badge-danger">✗ Offline</span>
              )}
            </div>
          </div>
          <div className="surface-card" style={{ padding: 24 }}>
            <div className="metric-label">OCR Engine</div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 4 }}>EasyOCR</div>
            <div className="metric-subtitle">English Model</div>
            <div style={{ marginTop: 12 }}>
              {health?.models?.ocr ? (
                <span className="badge badge-success">✓ Loaded</span>
              ) : loading ? (
                <span className="badge badge-info">Loading...</span>
              ) : (
                <span className="badge badge-danger">✗ Offline</span>
              )}
            </div>
          </div>
          <div className="surface-card" style={{ padding: 24 }}>
            <div className="metric-label">Background Removal</div>
            <div style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: 4 }}>rembg</div>
            <div className="metric-subtitle">U2-Net Model</div>
            <div style={{ marginTop: 12 }}>
              {health?.models?.rembg ? (
                <span className="badge badge-success">✓ Loaded</span>
              ) : loading ? (
                <span className="badge badge-info">Loading...</span>
              ) : (
                <span className="badge badge-danger">✗ Offline</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="surface-card" style={{ padding: 28 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
            🕐 Recent Activity
          </h2>
          <span className="badge badge-info">{recentLogs.length} entries</span>
        </div>
        <div>
          {recentLogs.length === 0 ? (
            <div style={{ textAlign: "center", padding: 32, color: "var(--on-surface-muted)" }}>
              <div style={{ fontSize: "2rem", marginBottom: 8 }}>📋</div>
              <div>No compliance checks recorded yet. Run a scan from the Monitor page.</div>
            </div>
          ) : (
            recentLogs.map((log: { planogram_name: string; compliance: number; detected: number; expected: number; alert_count: number; timestamp: string }, i: number) => (
              <div key={i} className="timeline-item" style={i === recentLogs.length - 1 ? { borderBottom: "none" } : {}}>
                <div
                  className="timeline-dot"
                  style={{
                    background:
                      log.compliance >= 90 ? "var(--primary)" :
                      log.compliance >= 70 ? "var(--warning)" :
                      "var(--danger)",
                  }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                    {log.planogram_name} — {log.compliance.toFixed(1)}% compliance
                  </div>
                  <div style={{ fontSize: "0.8rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                    {log.detected}/{log.expected} products detected • {log.alert_count} alert{log.alert_count !== 1 ? "s" : ""}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                    {log.timestamp ? new Date(log.timestamp).toLocaleString() : "—"}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
