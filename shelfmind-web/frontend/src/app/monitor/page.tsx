"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { API, apiUpload, apiFetch } from "@/lib/api";

interface ShelfCompliance {
  compliance: number;
  expected: number;
  detected: number;
  matched: number;
  issues: { type: string; product: string; expected_product?: string; position?: number; expected?: number; found?: number }[];
  revenue_at_risk: number;
}

interface ComplianceResult {
  overall_compliance: number;
  total_detected: number;
  total_expected: number;
  revenue_at_risk: number;
  shelf_compliance: Record<string, ShelfCompliance>;
  alerts: { type: string; shelf: number; product: string; priority: string; sku?: string }[];
  annotated_image: string;
}

import { useStore } from "@/lib/store";

export default function MonitorPage() {
  const store = useStore();
  const [planograms, setPlanograms] = useState<Record<string, unknown>>({});
  const [confidence, setConfidence] = useState(0.3);
  const [monitoring, setMonitoring] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanInterval, setScanInterval] = useState(5);
  const [source, setSource] = useState<"upload" | "webcam" | "ipwebcam">("upload");
  const [incidents, setIncidents] = useState<{ time: string; type: string; message: string }[]>([]);

  // ── Persisted state from global store ──
  const result = store.monitorResult as ComplianceResult | null;
  const setResult = store.setMonitorResult;
  const previewImage = store.monitorPreview;
  const setPreviewImage = store.setMonitorPreview;
  const selectedPlanogram = store.selectedPlanogram;
  const setSelectedPlanogram = store.setSelectedPlanogram;

  // IP Webcam
  const [ipWebcamUrl, setIpWebcamUrl] = useState("192.168.1.100:8080");
  const [ipConnected, setIpConnected] = useState(false);
  const ipImgRef = useRef<HTMLImageElement>(null);
  const ipIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const loadPlanograms = useCallback(async () => {
    try {
      const data = await apiFetch(API.planograms);
      setPlanograms(data.planograms || {});
      const keys = Object.keys(data.planograms || {});
      if (keys.length > 0 && !selectedPlanogram) setSelectedPlanogram(keys[0]);
    } catch { /* ignore */ }
  }, [selectedPlanogram, setSelectedPlanogram]);

  useEffect(() => { loadPlanograms(); }, [loadPlanograms]);

  // ── Webcam ──
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setSource("webcam");
    } catch (err) {
      alert("Webcam error: " + err);
    }
  };

  const stopWebcam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  const captureFrame = (): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (source === "ipwebcam" && ipImgRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = ipImgRef.current.naturalWidth;
        canvas.height = ipImgRef.current.naturalHeight;
        canvas.getContext("2d")?.drawImage(ipImgRef.current, 0, 0);
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
      } else if (source === "webcam" && videoRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0);
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
      } else if (previewImage) {
        fetch(previewImage).then((r) => r.blob()).then(resolve);
      } else {
        resolve(null);
      }
    });
  };

  // ── IP Webcam ──
  const connectIpWebcam = () => {
    const url = ipWebcamUrl.startsWith("http") ? ipWebcamUrl : `http://${ipWebcamUrl}`;
    const shotUrl = `${url}/shot.jpg`;

    const testImg = new Image();
    testImg.crossOrigin = "anonymous";
    testImg.onload = () => {
      setIpConnected(true);
      setSource("ipwebcam");
      if (ipIntervalRef.current) clearInterval(ipIntervalRef.current);
      ipIntervalRef.current = setInterval(() => {
        if (ipImgRef.current) {
          ipImgRef.current.src = `${shotUrl}?t=${Date.now()}`;
        }
      }, 500);
    };
    testImg.onerror = () => {
      alert(`Cannot connect to IP Webcam at ${url}.\nMake sure:\n1. IP Webcam app is running on your phone\n2. Phone and PC are on the same WiFi\n3. The IP address is correct`);
    };
    testImg.src = `${shotUrl}?t=${Date.now()}`;
  };

  const disconnectIpWebcam = () => {
    if (ipIntervalRef.current) {
      clearInterval(ipIntervalRef.current);
      ipIntervalRef.current = null;
    }
    setIpConnected(false);
  };

  // ── Single Compliance Check ──
  const runComplianceCheck = async () => {
    if (!selectedPlanogram) { alert("Select a planogram first."); return; }

    const blob = await captureFrame();
    if (!blob) { alert("No image available."); return; }

    setScanning(true);
    try {
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");
      formData.append("planogram_name", selectedPlanogram);
      formData.append("confidence", String(confidence));

      const res: ComplianceResult = await apiUpload(API.complianceCheck, formData);
      setResult(res);

      // Add to incidents
      const now = new Date().toLocaleTimeString();
      const newIncidents: { time: string; type: string; message: string }[] = [];

      res.alerts.forEach((a) => {
        newIncidents.push({
          time: now,
          type: a.type,
          message: `${a.type}: ${a.product} on Shelf ${a.shelf} (${a.priority})`,
        });
      });

      if (newIncidents.length > 0) {
        setIncidents((prev) => [...newIncidents, ...prev].slice(0, 20));
      } else {
        setIncidents((prev) => [{ time: now, type: "OK", message: `Scan complete — ${res.overall_compliance.toFixed(1)}% compliance` }, ...prev].slice(0, 20));
      }

      // Send notification for critical alerts
      const criticalAlerts = res.alerts.filter((a) => a.priority === "CRITICAL");
      if (criticalAlerts.length > 0) {
        try {
          const notifData = new FormData();
          notifData.append("title", `ShelfMind: ${criticalAlerts.length} Stockout(s)`);
          notifData.append("message", criticalAlerts.map((a) => `${a.product} - Shelf ${a.shelf}`).join("\n"));
          await apiUpload(API.notify, notifData);
        } catch { /* ignore */ }
      }
    } catch (err) {
      alert("Compliance check failed: " + (err as Error).message);
    } finally {
      setScanning(false);
    }
  };

  // ── Start/Stop Monitoring ──
  const startMonitoring = () => {
    if (!selectedPlanogram) { alert("Select a planogram first."); return; }
    setMonitoring(true);
    runComplianceCheck();
    intervalRef.current = setInterval(runComplianceCheck, scanInterval * 1000);
  };

  const stopMonitoring = () => {
    setMonitoring(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stopWebcam();
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      stopWebcam();
      disconnectIpWebcam();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreviewImage(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const getComplianceColor = (pct: number) => {
    if (pct >= 90) return "var(--primary)";
    if (pct >= 70) return "var(--warning)";
    return "var(--danger)";
  };

  const getComplianceBorder = (pct: number) => {
    if (pct >= 90) return "compliance-green";
    if (pct >= 70) return "compliance-yellow";
    return "compliance-red";
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">
          Live <span className="text-gradient">Monitor</span>
        </h1>
        <p className="page-subtitle">
          Real-time shelf compliance monitoring with automated alerts and incident tracking.
        </p>
      </div>

      {/* Config Bar */}
      <div className="surface-card" style={{ padding: 20, marginBottom: 24, display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center" }}>
        <div style={{ flex: "1 1 200px" }}>
          <label className="input-label">Planogram</label>
          <select className="select-field" value={selectedPlanogram} onChange={(e) => setSelectedPlanogram(e.target.value)}>
            <option value="">Select planogram...</option>
            {Object.keys(planograms).map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
        <div style={{ flex: "0 0 120px" }}>
          <label className="input-label">Confidence</label>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="range" min="0.1" max="0.9" step="0.05" value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} style={{ flex: 1, accentColor: "var(--primary)" }} />
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--primary-bright)" }}>{(confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div style={{ flex: "0 0 120px" }}>
          <label className="input-label">Scan Interval</label>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input type="range" min="3" max="30" step="1" value={scanInterval} onChange={(e) => setScanInterval(parseInt(e.target.value))} style={{ flex: 1, accentColor: "var(--primary)" }} />
            <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>{scanInterval}s</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {!monitoring ? (
            <button className="btn-primary" onClick={startMonitoring} disabled={!selectedPlanogram}>
              ▶️ Start Monitoring
            </button>
          ) : (
            <button className="btn-danger" onClick={stopMonitoring}>
              ⏹ Stop
            </button>
          )}
        </div>
      </div>

      <div className="grid-60-40">
        {/* Left: Live Feed + Shelves */}
        <div>
          {/* Source Tabs */}
          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            <button
              className={`segmented-btn ${source === "upload" ? "active" : ""}`}
              style={{ padding: "6px 14px", fontSize: "0.8rem", borderRadius: "var(--radius-sm)", background: source === "upload" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { setSource("upload"); stopWebcam(); disconnectIpWebcam(); }}
            >📁 Upload</button>
            <button
              className={`segmented-btn ${source === "webcam" ? "active" : ""}`}
              style={{ padding: "6px 14px", fontSize: "0.8rem", borderRadius: "var(--radius-sm)", background: source === "webcam" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { disconnectIpWebcam(); startWebcam(); }}
            >💻 Webcam</button>
            <button
              className={`segmented-btn ${source === "ipwebcam" ? "active" : ""}`}
              style={{ padding: "6px 14px", fontSize: "0.8rem", borderRadius: "var(--radius-sm)", background: source === "ipwebcam" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { stopWebcam(); setSource("ipwebcam"); }}
            >📱 Phone Camera</button>
          </div>

          {/* IP Webcam Connection */}
          {source === "ipwebcam" && !ipConnected && (
            <div className="surface-card" style={{ padding: 16, marginBottom: 12 }}>
              <div className="metric-label">IP Webcam Address</div>
              <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                <input
                  className="input-field"
                  placeholder="192.168.1.100:8080"
                  value={ipWebcamUrl}
                  onChange={(e) => setIpWebcamUrl(e.target.value)}
                  style={{ flex: 1 }}
                />
                <button className="btn-primary" onClick={connectIpWebcam} style={{ padding: "10px 20px" }}>
                  📲 Connect
                </button>
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--on-surface-muted)", marginTop: 6 }}>
                📱 Install <strong>IP Webcam</strong> Android app → Start Server → Enter IP shown
              </div>
            </div>
          )}

          {/* Live Feed */}
          <div className="camera-feed" style={{ marginBottom: 16 }}>
            {monitoring && <div className="live-badge"><span className="pulse-dot pulse-dot-red" /> LIVE</div>}
            {source === "ipwebcam" && ipConnected ? (
              <>
                {!monitoring && <div className="live-badge"><span className="pulse-dot pulse-dot-red" /> IP CAM</div>}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={ipImgRef}
                  alt="IP Webcam Feed"
                  crossOrigin="anonymous"
                  style={{ maxWidth: "100%", maxHeight: "70vh", objectFit: "contain" }}
                />
              </>
            ) : source === "webcam" ? (
              <video ref={videoRef} style={{ width: "100%", height: "100%", objectFit: "cover" }} autoPlay playsInline muted />
            ) : result?.annotated_image ? (
              <img src={`data:image/jpeg;base64,${result.annotated_image}`} alt="Annotated" />
            ) : previewImage ? (
              <img src={previewImage} alt="Preview" />
            ) : (
              <div style={{ textAlign: "center", color: "var(--on-surface-muted)" }}>
                <div style={{ fontSize: "3rem", marginBottom: 12 }}>📡</div>
                <div>Upload a shelf image or connect a camera</div>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
                <button className="btn-primary" style={{ marginTop: 16 }} onClick={() => fileRef.current?.click()}>
                  Choose Image
                </button>
              </div>
            )}
          </div>

          {source === "upload" && previewImage && !monitoring && (
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              <button className="btn-secondary" onClick={() => fileRef.current?.click()}>🔄 Change</button>
              <button className="btn-primary" onClick={runComplianceCheck} disabled={scanning || !selectedPlanogram}>
                {scanning ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Checking...</> : "🔍 Run Check"}
              </button>
              <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
            </div>
          )}

          {/* Shelf Compliance Cards */}
          {result && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {Object.entries(result.shelf_compliance).map(([shelfId, shelf]) => (
                <div key={shelfId} className={`shelf-card ${getComplianceBorder(shelf.compliance)}`}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span style={{ fontWeight: 600 }}>Shelf {shelfId}</span>
                    <span style={{ fontSize: "1.4rem", fontWeight: 800, color: getComplianceColor(shelf.compliance) }}>
                      {shelf.compliance.toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ fontSize: "0.82rem", color: shelf.compliance >= 90 ? "var(--primary)" : "var(--on-surface-dim)", fontWeight: 600, marginBottom: 8 }}>
                    {shelf.compliance >= 90 ? "Optimal Stock" : shelf.compliance >= 50 ? "LOW_STOCK" : "STOCKOUT"}
                  </div>
                  {/* Progress bar */}
                  <div style={{ height: 4, borderRadius: 2, background: "var(--surface-container)", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${shelf.compliance}%`, borderRadius: 2, background: getComplianceColor(shelf.compliance), transition: "width 0.5s ease" }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Global Metrics + Incident Log */}
        <div>
          {/* Global Compliance */}
          {result && (
            <div className="surface-card" style={{ padding: 24, marginBottom: 24 }}>
              <div className="metric-label">Global Compliance</div>
              <div className="metric-value" style={{ fontSize: "3rem", marginBottom: 16 }}>
                {result.overall_compliance.toFixed(1)}%
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <div className="metric-label">Detected</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>{result.total_detected}</div>
                </div>
                <div>
                  <div className="metric-label">Expected</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 700 }}>{result.total_expected}</div>
                </div>
                <div>
                  <div className="metric-label">Revenue at Risk</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 700, color: "var(--danger)" }}>
                    ₹{result.revenue_at_risk.toFixed(0)}
                  </div>
                </div>
                <div>
                  <div className="metric-label">Alerts</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 700, color: "var(--warning)" }}>
                    {result.alerts.length}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Incident Log */}
          <div className="surface-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>
              ⚡ Incident Log
            </h3>

            {incidents.length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "var(--on-surface-muted)" }}>
                Waiting for scan data...
              </div>
            ) : (
              <div style={{ maxHeight: 400, overflowY: "auto" }}>
                {incidents.map((inc, i) => (
                  <div key={i} className="timeline-item">
                    <div
                      className="timeline-dot"
                      style={{
                        background:
                          inc.type === "STOCKOUT" ? "var(--danger)" :
                          inc.type === "LOW_STOCK" ? "var(--warning)" :
                          inc.type === "MISPLACED" ? "var(--accent)" :
                          inc.type === "OK" ? "var(--primary)" :
                          "var(--on-surface-muted)",
                      }}
                    />
                    <div>
                      <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)" }}>{inc.time}</div>
                      <div style={{ fontSize: "0.85rem", fontWeight: 600, marginTop: 2 }}>{inc.message}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
