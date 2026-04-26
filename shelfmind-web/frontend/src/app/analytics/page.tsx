"use client";

import { useState, useEffect, useCallback } from "react";
import { API, apiFetch } from "@/lib/api";

interface ComplianceLog {
  id: number;
  planogram_name: string;
  compliance: number;
  detected: number;
  expected: number;
  revenue_risk: number;
  alert_count: number;
  timestamp: string;
}

interface AlertItem {
  id: number;
  alert_type: string;
  shelf_id: number;
  product_name: string;
  product_sku: string;
  priority: string;
  expected_count: number;
  found_count: number;
  revenue: number;
  timestamp: string;
}

export default function AnalyticsPage() {
  const [logs, setLogs] = useState<ComplianceLog[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [tab, setTab] = useState<"live" | "history">("live");

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [summary, setSummary] = useState<Record<string, any>>({});

  const loadData = useCallback(async () => {
    try {
      const [logsRes, alertsRes, summaryRes] = await Promise.all([
        apiFetch(API.complianceHistory),
        apiFetch(API.alerts),
        apiFetch(API.analyticsSummary).catch(() => ({})),
      ]);
      setLogs(logsRes.logs || []);
      setAlerts(alertsRes.alerts || []);
      setSummary(summaryRes || {});
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Computed metrics
  const totalScans = logs.length;
  const avgCompliance = logs.length > 0 ? logs.reduce((s, l) => s + l.compliance, 0) / logs.length : 0;
  const totalAlerts = alerts.length;
  const criticalAlerts = alerts.filter((a) => a.priority === "CRITICAL").length;
  const totalRevRisk = logs.reduce((s, l) => s + l.revenue_risk, 0);

  // Alert composition
  const alertTypes: Record<string, number> = {};
  alerts.forEach((a) => { alertTypes[a.alert_type] = (alertTypes[a.alert_type] || 0) + 1; });
  const totalAlertCount = alerts.length || 1;

  // Top offending products
  const productErrors: Record<string, number> = {};
  alerts.forEach((a) => {
    if (a.product_name) productErrors[a.product_name] = (productErrors[a.product_name] || 0) + 1;
  });
  const topOffenders = Object.entries(productErrors).sort((a, b) => b[1] - a[1]).slice(0, 5);

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <span style={{ fontSize: "0.72rem", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--on-surface-muted)" }}>
            Neural Intelligence Engine
          </span>
        </div>
        <h1 className="page-title">
          Compliance & <span className="text-gradient">Analytics</span>
        </h1>
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          <span className="badge badge-success">
            <span className="pulse-dot pulse-dot-green" /> Live Stream Active
          </span>
          <span className="badge badge-info">📅 Last 24 Hours</span>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid-metrics" style={{ marginBottom: 32 }}>
        <div className="metric-card">
          <div className="metric-label">Total Scans</div>
          <div className="metric-value">{totalScans.toLocaleString()}</div>
          <div className="metric-subtitle" style={{ color: "var(--primary)" }}>
            {summary?.total_scans ? `${summary.total_scans} total recorded` : "From compliance history"}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg Compliance</div>
          <div className="metric-value">{avgCompliance.toFixed(1)}%</div>
          <div className="metric-subtitle">
            {summary?.avg_compliance ? `Backend: ${summary.avg_compliance.toFixed(1)}%` : "Across all scans"}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Total Alerts</div>
          <div className="metric-value">{totalAlerts}</div>
          <div className="metric-subtitle" style={{ color: "var(--danger)" }}>
            ⚠ {criticalAlerts} Critical
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Revenue at Risk</div>
          <div className="metric-value metric-value-sm">₹{totalRevRisk.toLocaleString()}</div>
          <div className="metric-subtitle" style={{ color: "var(--accent)" }}>
            {summary?.products_registered ? `${summary.products_registered} products registered` : "🛡️ Shield Active"}
          </div>
        </div>
      </div>

      <div className="grid-60-40" style={{ marginBottom: 32 }}>
        {/* Left: Compliance Trend + Alert Composition */}
        <div>
          {/* Compliance Trend */}
          <div className="surface-card" style={{ padding: 24, marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Compliance Trend</h3>
                <div style={{ fontSize: "0.78rem", color: "var(--on-surface-muted)", marginTop: 2 }}>Real-time shelf accuracy across all SKUs</div>
              </div>
              <div className="segmented-toggle">
                <button className={`segmented-btn ${tab === "live" ? "active" : ""}`} style={{ padding: "6px 14px", fontSize: "0.78rem" }} onClick={() => setTab("live")}>Live</button>
                <button className={`segmented-btn ${tab === "history" ? "active" : ""}`} style={{ padding: "6px 14px", fontSize: "0.78rem" }} onClick={() => setTab("history")}>History</button>
              </div>
            </div>

            {/* Simple chart visualization */}
            <div style={{ height: 200, display: "flex", alignItems: "end", gap: 4, padding: "20px 0" }}>
              {(logs.length > 0 ? logs.slice(-20) : [82, 76, 88, 91, 85, 79, 93, 87, 90, 84, 95, 89].map((c, i) => ({ compliance: c, timestamp: `${8 + i}:00` }))).map((log, i) => {
                const l = log as ComplianceLog & { compliance: number };
                const height = Math.max(10, (l.compliance / 100) * 160);
                return (
                  <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                    <div
                      style={{
                        width: "100%",
                        height: height,
                        borderRadius: "4px 4px 0 0",
                        background: `linear-gradient(to top, var(--primary), var(--secondary))`,
                        opacity: 0.6 + (i / 20) * 0.4,
                        transition: "height 0.3s ease",
                      }}
                      title={`${l.compliance.toFixed(1)}%`}
                    />
                  </div>
                );
              })}
            </div>
          </div>

          {/* Alert Composition */}
          <div className="surface-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>Alert Composition</h3>

            <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
              {/* Donut Chart (CSS) */}
              <div style={{ position: "relative", width: 160, height: 160 }}>
                <svg viewBox="0 0 36 36" style={{ transform: "rotate(-90deg)", width: "100%", height: "100%" }}>
                  {Object.entries(alertTypes).reduce<{ elements: React.ReactNode[]; offset: number }>((acc, [type, count], i) => {
                    const pct = (count / totalAlertCount) * 100;
                    const colors = ["var(--danger)", "var(--warning)", "var(--accent)", "var(--secondary)"];
                    acc.elements.push(
                      <circle key={type} cx="18" cy="18" r="14" fill="none" strokeWidth="4"
                        stroke={colors[i % colors.length]}
                        strokeDasharray={`${pct} ${100 - pct}`}
                        strokeDashoffset={`-${acc.offset}`}
                      />
                    );
                    acc.offset += pct;
                    return acc;
                  }, { elements: [], offset: 0 }).elements}
                </svg>
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ fontSize: "1.5rem", fontWeight: 800 }}>{totalAlerts}</div>
                  <div style={{ fontSize: "0.65rem", color: "var(--on-surface-muted)", textTransform: "uppercase" }}>Total</div>
                </div>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {Object.entries(alertTypes).map(([type, count], i) => {
                const colors = ["var(--danger)", "var(--warning)", "var(--accent)", "var(--secondary)"];
                return (
                  <div key={type} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: colors[i % colors.length] }} />
                    <span style={{ flex: 1, fontSize: "0.85rem" }}>{type.replace("_", " ")}</span>
                    <span style={{ fontWeight: 700, fontSize: "0.85rem" }}>{((count / totalAlertCount) * 100).toFixed(0)}%</span>
                  </div>
                );
              })}
              {Object.keys(alertTypes).length === 0 && (
                <div style={{ textAlign: "center", color: "var(--on-surface-muted)", padding: 16 }}>No alerts recorded yet</div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Top Offenders + Alert History */}
        <div>
          {/* Top Offending Products */}
          <div className="surface-card" style={{ padding: 24, marginBottom: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>Top Offending Products</h3>
            {topOffenders.length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "var(--on-surface-muted)" }}>No data yet</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {topOffenders.map(([name, count], i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{ width: 32, height: 32, borderRadius: "var(--radius-sm)", background: "var(--surface-container)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.8rem" }}>
                      {i + 1}
                    </div>
                    <span style={{ flex: 1, fontSize: "0.85rem", fontWeight: 500 }}>{name}</span>
                    <span className="badge badge-danger">{count} Errors</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Alert History Table */}
          <div className="surface-card" style={{ padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Alert History</h3>
              <button className="btn-secondary" style={{ padding: "6px 14px", fontSize: "0.78rem" }} onClick={loadData}>
                🔄 Refresh
              </button>
            </div>

            {alerts.length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "var(--on-surface-muted)" }}>No alerts recorded yet</div>
            ) : (
              <div style={{ maxHeight: 350, overflowY: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--glass-border)" }}>
                      <th style={{ textAlign: "left", padding: "8px 4px", color: "var(--on-surface-muted)", fontWeight: 600, fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>Time</th>
                      <th style={{ textAlign: "left", padding: "8px 4px", color: "var(--on-surface-muted)", fontWeight: 600, fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>Type</th>
                      <th style={{ textAlign: "left", padding: "8px 4px", color: "var(--on-surface-muted)", fontWeight: 600, fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>Product</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.slice(0, 15).map((a) => (
                      <tr key={a.id} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                        <td style={{ padding: "8px 4px", color: "var(--on-surface-muted)" }}>
                          {a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : "—"}
                        </td>
                        <td style={{ padding: "8px 4px" }}>
                          <span className={`badge ${a.alert_type === "STOCKOUT" ? "badge-danger" : a.alert_type === "LOW_STOCK" ? "badge-warning" : "badge-purple"}`} style={{ fontSize: "0.7rem" }}>
                            {a.alert_type}
                          </span>
                        </td>
                        <td style={{ padding: "8px 4px" }}>{a.product_name}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div style={{ textAlign: "center", marginTop: 12 }}>
              <button className="btn-secondary" style={{ padding: "6px 14px", fontSize: "0.78rem", color: "var(--primary)" }}>
                View All Activity Logs →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
