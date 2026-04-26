"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { API, apiUpload, apiFetch } from "@/lib/api";

interface ShelfProduct {
  position: number;
  sku: string;
  name: string;
  confidence: number;
  bbox: number[];
}

interface ShelfLevel {
  level: number;
  product_count: number;
  products: ShelfProduct[];
}

interface DetectResult {
  n_shelves: number;
  total_products: number;
  shelves: ShelfLevel[];
  annotated_image: string;
}

interface ManualShelf {
  level: number;
  products: { sku: string; name: string; facing: number }[];
}

interface Planogram {
  name: string;
  created_at?: string;
  shelves?: ShelfLevel[];
  [key: string]: unknown;
}

export default function PlanogramPage() {
  const [tab, setTab] = useState<"auto" | "manual">("auto");
  const [detecting, setDetecting] = useState(false);
  const [detectResult, setDetectResult] = useState<DetectResult | null>(null);
  const [planograms, setPlanograms] = useState<Record<string, Planogram>>({});
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Manual mode
  const [manualShelves, setManualShelves] = useState<ManualShelf[]>([
    { level: 1, products: [] },
  ]);
  const [products, setProducts] = useState<{ sku: string; name: string }[]>([]);

  const loadPlanograms = useCallback(async () => {
    try {
      const data = await apiFetch(API.planograms);
      setPlanograms(data.planograms || {});
    } catch { /* ignore */ }
  }, []);

  const loadProducts = useCallback(async () => {
    try {
      const data = await apiFetch(API.products);
      setProducts((data.products || []).map((p: { sku: string; name: string }) => ({ sku: p.sku, name: p.name })));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadPlanograms();
    loadProducts();
  }, [loadPlanograms, loadProducts]);

  // ── Auto-Detect ──
  const handleDetect = async () => {
    if (!fileRef.current?.files?.length && !previewImage) {
      alert("Please upload a shelf image.");
      return;
    }
    setDetecting(true);
    setDetectResult(null);

    try {
      const formData = new FormData();
      if (fileRef.current?.files?.length) {
        formData.append("image", fileRef.current.files[0]);
      } else if (previewImage) {
        const blob = await (await fetch(previewImage)).blob();
        formData.append("image", blob, "shelf.jpg");
      }
      formData.append("confidence", "0.25");
      const result: DetectResult = await apiUpload(API.planogramAutoDetect, formData);
      setDetectResult(result);
    } catch (err) {
      alert("Detection failed: " + (err as Error).message);
    } finally {
      setDetecting(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => setPreviewImage(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  // ── Save Planogram ──
  const handleSave = async () => {
    const name = saveName.trim();
    if (!name) { alert("Enter a planogram name."); return; }

    setSaving(true);
    try {
      const formData = new FormData();
      formData.append("name", name);

      const planData = tab === "auto" && detectResult
        ? { shelves: detectResult.shelves, n_shelves: detectResult.n_shelves, total_products: detectResult.total_products }
        : {
            shelves: manualShelves.map((s) => ({
              level: s.level,
              product_count: s.products.reduce((sum, p) => sum + p.facing, 0),
              products: s.products.flatMap((p) =>
                Array.from({ length: p.facing }, (_, i) => ({
                  position: i,
                  sku: p.sku,
                  name: p.name,
                  confidence: 1.0,
                  bbox: [0, 0, 0, 0],
                }))
              ),
            })),
            n_shelves: manualShelves.length,
            total_products: manualShelves.reduce((sum, s) => sum + s.products.reduce((ss, p) => ss + p.facing, 0), 0),
          };

      formData.append("data", JSON.stringify(planData));

      if (previewImage) {
        const blob = await (await fetch(previewImage)).blob();
        formData.append("image", blob, "reference.jpg");
      }

      await apiUpload(API.planograms, formData);
      alert(`✅ Planogram "${name}" saved!`);
      setSaveName("");
      loadPlanograms();
    } catch (err) {
      alert("Save failed: " + (err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  // ── Delete Planogram ──
  const handleDeletePlanogram = async (name: string) => {
    if (!confirm(`Delete planogram "${name}"?`)) return;
    try {
      await apiFetch(API.planogramDelete(name), { method: "DELETE" });
      loadPlanograms();
    } catch { /* ignore */ }
  };

  // ── Manual Shelf Helpers ──
  const addShelf = () => setManualShelves([...manualShelves, { level: manualShelves.length + 1, products: [] }]);
  const removeShelf = (i: number) => setManualShelves(manualShelves.filter((_, idx) => idx !== i));
  const addProductToShelf = (shelfIdx: number, sku: string, name: string) => {
    const updated = [...manualShelves];
    updated[shelfIdx].products.push({ sku, name, facing: 1 });
    setManualShelves(updated);
  };
  const removeProductFromShelf = (shelfIdx: number, prodIdx: number) => {
    const updated = [...manualShelves];
    updated[shelfIdx].products.splice(prodIdx, 1);
    setManualShelves(updated);
  };
  const updateFacing = (shelfIdx: number, prodIdx: number, facing: number) => {
    const updated = [...manualShelves];
    updated[shelfIdx].products[prodIdx].facing = Math.max(1, facing);
    setManualShelves(updated);
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">
          Planogram <span className="text-gradient">Creator</span>
        </h1>
        <p className="page-subtitle">
          Architect your retail floor with neural precision. Design, detect, and deploy optimized shelf layouts.
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="segmented-toggle" style={{ marginBottom: 28 }}>
        <button className={`segmented-btn ${tab === "auto" ? "active" : ""}`} onClick={() => setTab("auto")}>
          🤖 Auto-Detect
        </button>
        <button className={`segmented-btn ${tab === "manual" ? "active" : ""}`} onClick={() => setTab("manual")}>
          ✏️ Manual Editor
        </button>
      </div>

      <div className="grid-60-40">
        {/* Left Column */}
        <div>
          {tab === "auto" ? (
            <>
              {/* Upload Area */}
              <div className="camera-feed" style={{ marginBottom: 16 }}>
                {previewImage ? (
                  <img src={detectResult?.annotated_image ? `data:image/jpeg;base64,${detectResult.annotated_image}` : previewImage} alt="Shelf" />
                ) : (
                  <div style={{ textAlign: "center", color: "var(--on-surface-muted)" }}>
                    <div style={{ fontSize: "3rem", marginBottom: 12 }}>🏪</div>
                    <div>Upload a shelf image for auto-detection</div>
                    <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
                    <button className="btn-primary" style={{ marginTop: 16 }} onClick={() => fileRef.current?.click()}>
                      Choose Shelf Image
                    </button>
                  </div>
                )}
              </div>

              <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
                {previewImage && (
                  <button className="btn-secondary" onClick={() => fileRef.current?.click()}>🔄 Change</button>
                )}
                <button className="btn-primary" onClick={handleDetect} disabled={detecting || !previewImage}>
                  {detecting ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Processing...</> : "⚡ Process Frame"}
                </button>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
              </div>

              {/* Detection Results */}
              {detectResult && (
                <div className="surface-card" style={{ padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                    <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>🧠 Neural Detection Breakdown</h3>
                    <span className="badge badge-info">{detectResult.n_shelves} shelves detected</span>
                  </div>

                  {detectResult.shelves.map((shelf) => (
                    <div key={shelf.level} className="shelf-card" style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                        <div>
                          <span style={{ fontWeight: 700, marginRight: 8 }}>Shelf {shelf.level}</span>
                          <span style={{ fontSize: "0.78rem", color: "var(--on-surface-muted)" }}>
                            {shelf.product_count} products
                          </span>
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {shelf.products.map((p, i) => (
                          <span
                            key={i}
                            className={`badge ${p.sku === "UNKNOWN" ? "badge-danger" : "badge-success"}`}
                          >
                            {p.name} {p.confidence > 0 && `${(p.confidence * 100).toFixed(0)}%`}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            /* Manual Editor */
            <div>
              {manualShelves.map((shelf, si) => (
                <div key={si} className="surface-card" style={{ padding: 24, marginBottom: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <h4 style={{ fontWeight: 700 }}>Shelf Level {shelf.level}</h4>
                    {manualShelves.length > 1 && (
                      <button style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer" }} onClick={() => removeShelf(si)}>
                        🗑️ Remove
                      </button>
                    )}
                  </div>

                  {/* Products on this shelf */}
                  {shelf.products.map((p, pi) => (
                    <div
                      key={pi}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        padding: "8px 12px",
                        borderRadius: "var(--radius-sm)",
                        background: "var(--surface-container)",
                        marginBottom: 8,
                      }}
                    >
                      <span style={{ flex: 1, fontSize: "0.85rem" }}>
                        <span className="badge badge-success" style={{ marginRight: 8 }}>{p.name}</span>
                      </span>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: "0.75rem", color: "var(--on-surface-muted)" }}>Facing:</span>
                        <button className="btn-secondary" style={{ padding: "4px 8px", fontSize: "0.8rem" }} onClick={() => updateFacing(si, pi, p.facing - 1)}>−</button>
                        <span style={{ fontWeight: 700, minWidth: 20, textAlign: "center" }}>{p.facing}</span>
                        <button className="btn-secondary" style={{ padding: "4px 8px", fontSize: "0.8rem" }} onClick={() => updateFacing(si, pi, p.facing + 1)}>+</button>
                      </div>
                      <button style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer" }} onClick={() => removeProductFromShelf(si, pi)}>✕</button>
                    </div>
                  ))}

                  {/* Add product dropdown */}
                  <select
                    className="select-field"
                    style={{ marginTop: 8 }}
                    value=""
                    onChange={(e) => {
                      const prod = products.find((p) => p.sku === e.target.value);
                      if (prod) addProductToShelf(si, prod.sku, prod.name);
                    }}
                  >
                    <option value="">+ Add SKU...</option>
                    {products.map((p) => (
                      <option key={p.sku} value={p.sku}>{p.name} ({p.sku})</option>
                    ))}
                  </select>
                </div>
              ))}

              <button className="btn-secondary" style={{ width: "100%", marginBottom: 16 }} onClick={addShelf}>
                ➕ Add Shelf Level
              </button>
            </div>
          )}
        </div>

        {/* Right Column: Save + History */}
        <div>
          {/* Save */}
          <div className="surface-card" style={{ padding: 24, marginBottom: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>💾 Save Planogram</h3>
            <div>
              <label className="input-label">Planogram Name</label>
              <input className="input-field" placeholder="e.g. Main Aisle Q4-2025" value={saveName} onChange={(e) => setSaveName(e.target.value)} />
            </div>
            <button
              className="btn-primary"
              style={{ width: "100%", marginTop: 16, justifyContent: "center" }}
              onClick={handleSave}
              disabled={saving || !saveName.trim()}
            >
              {saving ? "Saving..." : "💾 Save Planogram"}
            </button>
          </div>

          {/* Deployment History */}
          <div className="surface-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 16 }}>📋 Deployment History</h3>

            {Object.keys(planograms).length === 0 ? (
              <div style={{ textAlign: "center", padding: 24, color: "var(--on-surface-muted)" }}>
                <div style={{ fontSize: "2rem", marginBottom: 8 }}>📋</div>
                No planograms saved yet.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {Object.entries(planograms).map(([name, plan]) => (
                  <div key={name} className="shelf-card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                      <div>
                        <div style={{ fontWeight: 700, fontSize: "0.95rem" }}>{name}</div>
                        <div style={{ fontSize: "0.75rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                          {plan.created_at ? new Date(plan.created_at).toLocaleDateString() : "No date"}
                        </div>
                        <div style={{ fontSize: "0.78rem", color: "var(--on-surface-dim)", marginTop: 4 }}>
                          Shelves: {plan.shelves?.length || "?"} • Products: {plan.shelves?.reduce((s: number, sh: ShelfLevel) => s + sh.product_count, 0) || "?"}
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <span className="badge badge-success">Active</span>
                        <button
                          style={{ background: "none", border: "none", cursor: "pointer", color: "var(--danger)" }}
                          onClick={() => handleDeletePlanogram(name)}
                        >🗑️</button>
                      </div>
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
