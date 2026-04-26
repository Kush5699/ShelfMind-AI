"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { API, apiUpload, apiFetch } from "@/lib/api";

interface ScanResult {
  cropped_image: string;
  original_image: string;
  embedding: number[];
  augmentation_views: string[];
  augmentation_images: string[];
  embedding_dim: number;
}

interface BulkProduct {
  crop_image: string;
  bbox: number[];
  confidence: number;
  embedding: number[];
  ocr_text: string;
  count: number;
}

interface BulkResult {
  total_detected: number;
  unique_products: number;
  products: BulkProduct[];
  annotated_image: string;
}

interface Product {
  sku: string;
  name: string;
  category: string;
  price: number;
  image_url?: string;
  has_embedding: boolean;
}

import { useStore } from "@/lib/store";

export default function ScannerPage() {
  const store = useStore();
  const [mode, setMode] = useState<"single" | "bulk">("single");
  const [source, setSource] = useState<"upload" | "webcam" | "ipwebcam">("upload");

  // IP Webcam
  const [ipWebcamUrl, setIpWebcamUrl] = useState("192.168.1.100:8080");
  const [ipConnected, setIpConnected] = useState(false);
  const ipImgRef = useRef<HTMLImageElement>(null);
  const ipIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [scanning, scanStatus, scanError] = [store.scanning, store.scanStatus, store.scanError];
  const [bulkConfidence, setBulkConfidence] = useState(0.45);
  const [bulkSimilarity, setBulkSimilarity] = useState(0.65);
  const [registering, setRegistering] = useState(false);

  // ── Use global store for persisted state ──
  const scanResult = store.scanResult as ScanResult | null;
  const setScanResult = store.setScanResult;
  const bulkResult = store.bulkResult as BulkResult | null;
  const setBulkResult = store.setBulkResult;
  const products = store.products;
  const previewImage = store.scannerPreview;
  const setPreviewImage = store.setScannerPreview;

  // Form state from store
  const [productName, setProductName] = useState(store.scannerFormState.name);
  const [productBarcode, setProductBarcode] = useState(store.scannerFormState.barcode);
  const [productCategory, setProductCategory] = useState(store.scannerFormState.category);
  const [productPrice, setProductPrice] = useState(store.scannerFormState.price);

  // Sync form state back to store on change
  const syncFormToStore = useCallback(() => {
    store.setScannerFormState({ name: productName, barcode: productBarcode, category: productCategory, price: productPrice });
  }, [store, productName, productBarcode, productCategory, productPrice]);

  // Voice
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Load products from API (uses global store)
  const loadProducts = store.loadProducts;

  // Load on mount (only if not already loaded)
  useEffect(() => { if (!store.productsLoaded) loadProducts(); }, [loadProducts, store.productsLoaded]);

  // Save form state on unmount
  useEffect(() => {
    return () => { syncFormToStore(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productName, productBarcode, productCategory, productPrice]);

  // ── File Upload ──
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setPreviewImage(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

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
      alert("Could not access webcam: " + err);
    }
  };

  const captureWebcam = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0);
    setPreviewImage(canvas.toDataURL("image/jpeg", 0.92));
    stopWebcam();
  };

  const stopWebcam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  // ── IP Webcam ──
  const connectIpWebcam = () => {
    const url = ipWebcamUrl.startsWith("http") ? ipWebcamUrl : `http://${ipWebcamUrl}`;
    const shotUrl = `${url}/shot.jpg`;

    // Test connection with a single fetch
    const testImg = new Image();
    testImg.crossOrigin = "anonymous";
    testImg.onload = () => {
      setIpConnected(true);
      setSource("ipwebcam");
      // Start refreshing snapshot every 500ms for live preview
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

  const captureIpWebcam = () => {
    if (!ipImgRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = ipImgRef.current.naturalWidth;
    canvas.height = ipImgRef.current.naturalHeight;
    canvas.getContext("2d")?.drawImage(ipImgRef.current, 0, 0);
    setPreviewImage(canvas.toDataURL("image/jpeg", 0.92));
    disconnectIpWebcam();
  };

  const disconnectIpWebcam = () => {
    if (ipIntervalRef.current) {
      clearInterval(ipIntervalRef.current);
      ipIntervalRef.current = null;
    }
    setIpConnected(false);
  };

  // ── Scan (runs in global store — survives tab switches) ──
  const handleScan = async () => {
    if (!previewImage && !fileInputRef.current?.files?.length) {
      alert("Please upload or capture an image first.");
      return;
    }

    const formData = new FormData();
    if (fileInputRef.current?.files?.length) {
      formData.append("image", fileInputRef.current.files[0]);
    } else if (previewImage) {
      const blob = await (await fetch(previewImage)).blob();
      formData.append("image", blob, "capture.jpg");
    }

    if (mode === "bulk") {
      formData.append("similarity_threshold", String(bulkSimilarity));
      formData.append("confidence", String(bulkConfidence));
    }

    store.startScan(formData, mode);
  };

  // ── Register Product ──
  const handleRegister = async () => {
    if (!productName.trim()) {
      alert("Please enter a product name.");
      return;
    }

    setRegistering(true);
    try {
      const formData = new FormData();
      formData.append("name", productName);
      formData.append("category", productCategory);
      formData.append("price", productPrice || "0");
      formData.append("barcode", productBarcode);

      if (scanResult?.embedding) {
        formData.append("embedding", JSON.stringify(scanResult.embedding));
      }

      // Use cropped image or original
      if (scanResult?.cropped_image) {
        const blob = await (await fetch(`data:image/jpeg;base64,${scanResult.cropped_image}`)).blob();
        formData.append("image", blob, "product.jpg");
      } else if (previewImage) {
        const blob = await (await fetch(previewImage)).blob();
        formData.append("image", blob, "product.jpg");
      }

      await apiUpload(API.products, formData);
      alert(`✅ ${productName} registered successfully!`);

      // Reset form
      setProductName("");
      setProductBarcode("");
      setProductCategory("Other");
      setProductPrice("");
      setScanResult(null);
      setPreviewImage(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      loadProducts();
    } catch (err) {
      alert("Registration failed: " + (err as Error).message);
    } finally {
      setRegistering(false);
    }
  };

  // ── Bulk Register ──
  const handleBulkRegister = async (product: BulkProduct, name: string) => {
    try {
      const formData = new FormData();
      formData.append("name", name);
      formData.append("category", "Other");
      formData.append("price", "0");
      formData.append("barcode", "");
      formData.append("embedding", JSON.stringify(product.embedding));

      const blob = await (await fetch(`data:image/jpeg;base64,${product.crop_image}`)).blob();
      formData.append("image", blob, "product.jpg");

      await apiUpload(API.products, formData);
      alert(`✅ ${name} registered!`);
      loadProducts();
    } catch (err) {
      alert("Registration failed: " + (err as Error).message);
    }
  };

  // ── Voice ──
  const toggleVoice = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition = (window as unknown as { SpeechRecognition?: typeof window.SpeechRecognition; webkitSpeechRecognition?: typeof window.SpeechRecognition }).SpeechRecognition || (window as unknown as { webkitSpeechRecognition?: typeof window.SpeechRecognition }).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-IN";

    recognition.onresult = async (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      setIsRecording(false);

      try {
        const formData = new FormData();
        formData.append("transcript", transcript);
        const result = await apiUpload(API.productVoice, formData);
        if (result.parsed) {
          setProductName(result.parsed.name);
          setProductCategory(result.parsed.category);
          if (result.parsed.price > 0) setProductPrice(String(result.parsed.price));
        }
      } catch { /* ignore */ }
    };

    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);

    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  };

  // ── Delete Product ──
  const handleDelete = async (sku: string) => {
    if (!confirm("Delete this product?")) return;
    try {
      await apiFetch(API.productDelete(sku), { method: "DELETE" });
      loadProducts();
    } catch { /* ignore */ }
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">
          Product <span className="text-gradient">Scanner</span>
        </h1>
        <p className="page-subtitle">
          Scan, identify, and register products using AI-powered vision, OCR, and barcode detection.
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="segmented-toggle" style={{ marginBottom: 28 }}>
        <button className={`segmented-btn ${mode === "single" ? "active" : ""}`} onClick={() => { setMode("single"); setBulkResult(null); }}>
          📷 Single Product
        </button>
        <button className={`segmented-btn ${mode === "bulk" ? "active" : ""}`} onClick={() => { setMode("bulk"); setScanResult(null); }}>
          🏪 Bulk Shelf
        </button>
      </div>

      {/* Bulk Mode: Detection Controls */}
      {mode === "bulk" && (
        <div className="surface-card" style={{ padding: "16px 20px", marginBottom: 20, display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Detection Confidence */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="input-label" style={{ margin: 0, whiteSpace: "nowrap", minWidth: 140 }}>Detection Confidence</span>
            <input
              type="range" min="0.2" max="0.8" step="0.05"
              value={bulkConfidence}
              onChange={(e) => setBulkConfidence(parseFloat(e.target.value))}
              style={{ flex: 1, minWidth: 80, accentColor: "var(--primary)" }}
            />
            <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--primary-bright)", minWidth: 40, textAlign: "right" }}>
              {(bulkConfidence * 100).toFixed(0)}%
            </span>
          </div>
          {/* Similarity Threshold */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="input-label" style={{ margin: 0, whiteSpace: "nowrap", minWidth: 140 }}>Similarity Grouping</span>
            <input
              type="range" min="0.4" max="0.95" step="0.05"
              value={bulkSimilarity}
              onChange={(e) => setBulkSimilarity(parseFloat(e.target.value))}
              style={{ flex: 1, minWidth: 80, accentColor: "var(--accent)" }}
            />
            <span style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent)", minWidth: 40, textAlign: "right" }}>
              {(bulkSimilarity * 100).toFixed(0)}%
            </span>
          </div>
          <div style={{ fontSize: "0.7rem", color: "var(--on-surface-muted)", lineHeight: 1.5 }}>
            🎯 <strong>Confidence:</strong> Higher = ignore weak detections &nbsp;|&nbsp;
            📦 <strong>Similarity:</strong> Lower = more unique products (less grouping)
          </div>
        </div>
      )}

      <div className="grid-60-40">
        {/* Left: Camera + Results */}
        <div>
          {/* Source Tabs */}
          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <button
              className={`segmented-btn ${source === "upload" ? "active" : ""}`}
              style={{ padding: "8px 16px", fontSize: "0.82rem", borderRadius: "var(--radius-sm)", background: source === "upload" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { setSource("upload"); stopWebcam(); disconnectIpWebcam(); }}
            >📁 Upload</button>
            <button
              className={`segmented-btn ${source === "webcam" ? "active" : ""}`}
              style={{ padding: "8px 16px", fontSize: "0.82rem", borderRadius: "var(--radius-sm)", background: source === "webcam" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { disconnectIpWebcam(); startWebcam(); }}
            >💻 Webcam</button>
            <button
              className={`segmented-btn ${source === "ipwebcam" ? "active" : ""}`}
              style={{ padding: "8px 16px", fontSize: "0.82rem", borderRadius: "var(--radius-sm)", background: source === "ipwebcam" ? "var(--primary-container)" : "var(--surface-container)" }}
              onClick={() => { stopWebcam(); setSource("ipwebcam"); }}
            >📱 Phone Camera</button>
          </div>

          {/* IP Webcam Connection */}
          {source === "ipwebcam" && !ipConnected && (
            <div className="surface-card" style={{ padding: 20, marginBottom: 16 }}>
              <div className="metric-label">IP Webcam Address</div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <input
                  className="input-field"
                  placeholder="192.168.1.100:8080"
                  value={ipWebcamUrl}
                  onChange={(e) => setIpWebcamUrl(e.target.value)}
                  style={{ flex: 1 }}
                />
                <button className="btn-primary" onClick={connectIpWebcam}>
                  📲 Connect
                </button>
              </div>
              <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)", marginTop: 8 }}>
                📱 Install <strong>IP Webcam</strong> app on your Android phone → Start Server → Enter IP shown
              </div>
            </div>
          )}

          {/* Camera / Upload Area */}
          <div className="camera-feed" style={{ marginBottom: 20 }}>
            {source === "ipwebcam" && ipConnected ? (
              <>
                <div className="live-badge">
                  <span className="pulse-dot pulse-dot-red" /> IP CAM LIVE
                </div>
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
            ) : previewImage ? (
              <img src={previewImage} alt="Preview" />
            ) : (
              <div style={{ textAlign: "center", color: "var(--on-surface-muted)" }}>
                <div style={{ fontSize: "3rem", marginBottom: 12 }}>📸</div>
                <div style={{ fontSize: "0.9rem" }}>Upload or capture an image</div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                />
                <button
                  className="btn-primary"
                  style={{ marginTop: 16 }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose Image
                </button>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
            {source === "upload" && previewImage && (
              <button className="btn-secondary" onClick={() => fileInputRef.current?.click()}>
                🔄 Change Image
              </button>
            )}
            {source === "webcam" && (
              <button className="btn-primary" onClick={captureWebcam}>📸 Capture</button>
            )}
            {source === "ipwebcam" && ipConnected && (
              <>
                <button className="btn-primary" onClick={captureIpWebcam}>📸 Capture</button>
                <button className="btn-danger" onClick={() => { disconnectIpWebcam(); setSource("ipwebcam"); }} style={{ padding: "12px 20px" }}>⏹ Disconnect</button>
              </>
            )}
            <button className="btn-primary" onClick={handleScan} disabled={scanning || (!previewImage && !fileInputRef.current?.files?.length)}>
              {scanning ? (
                <><span className="spinner" style={{ width: 16, height: 16 }} /> {scanStatus || "Processing..."}</>
              ) : (
                <>{mode === "single" ? "🔍 Scan Product" : "🔍 Scan Shelf"}</>
              )}
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: "none" }} />
          </div>

          {/* Scan Error */}
          {scanError && (
            <div style={{ padding: "12px 16px", background: "rgba(255,67,67,0.1)", border: "1px solid rgba(255,67,67,0.3)", borderRadius: "var(--radius-md)", color: "#ff6b6b", fontSize: "0.85rem", marginBottom: 16 }}>
              ❌ Scan failed: {scanError}
            </div>
          )}

          {/* Single Scan Results */}
          {scanResult && mode === "single" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Auto-detect row */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div className="surface-card" style={{ padding: 16 }}>
                  <div className="metric-label">✂️ AI Auto-Crop (rembg)</div>
                  {scanResult.cropped_image && (
                    <img
                      src={`data:image/jpeg;base64,${scanResult.cropped_image}`}
                      alt="Cropped"
                      style={{ width: "100%", height: 80, objectFit: "contain", marginTop: 8, borderRadius: 8 }}
                    />
                  )}
                </div>
                <div className="surface-card" style={{ padding: 16 }}>
                  <div className="metric-label">🧬 DINOv2 Embedding</div>
                  <div style={{ fontSize: "1.4rem", fontWeight: 800, marginTop: 8, color: "var(--primary-bright)" }}>
                    {scanResult.embedding_dim}d
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)", marginTop: 4 }}>
                    {scanResult.augmentation_views?.length || 10} augmented views averaged
                  </div>
                </div>
              </div>

              {/* Augmented Views */}
              {scanResult.augmentation_images && scanResult.augmentation_images.length > 0 && (
                <div className="surface-card" style={{ padding: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div className="metric-label" style={{ margin: 0 }}>🔄 Augmented Projections</div>
                    <span className="badge badge-info">{scanResult.augmentation_views?.length || 10} views</span>
                  </div>
                  <div className="aug-grid">
                    {scanResult.augmentation_images.map((img, i) => (
                      <div key={i} className="aug-thumb" title={scanResult.augmentation_views?.[i] || `View ${i+1}`}>
                        <img
                          src={`data:image/jpeg;base64,${img}`}
                          alt={scanResult.augmentation_views?.[i] || `View ${i+1}`}
                          style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 6 }}
                        />
                        <div style={{ position: "absolute", bottom: 2, left: 0, right: 0, textAlign: "center", fontSize: "0.55rem", color: "var(--on-surface-muted)", background: "rgba(0,0,0,0.5)", borderRadius: "0 0 6px 6px", padding: "2px 0" }}>
                          {scanResult.augmentation_views?.[i]}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Bulk Scan Results */}
          {bulkResult && mode === "bulk" && (
            <div>
              <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                <span className="badge badge-success">🎯 {bulkResult.total_detected} detected</span>
                <span className="badge badge-info">📦 {bulkResult.unique_products} unique</span>
              </div>

              {bulkResult.annotated_image && (
                <div className="surface-card" style={{ padding: 4, marginBottom: 16 }}>
                  <img
                    src={`data:image/jpeg;base64,${bulkResult.annotated_image}`}
                    alt="Annotated shelf"
                    style={{ width: "100%", borderRadius: "var(--radius-lg)" }}
                  />
                </div>
              )}

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12 }}>
                {bulkResult.products.map((p, i) => (
                  <div key={i} className="surface-card" style={{ padding: 12 }}>
                    <img
                      src={`data:image/jpeg;base64,${p.crop_image}`}
                      alt={`Product ${i + 1}`}
                      style={{ width: "100%", height: 100, objectFit: "contain", borderRadius: 8, marginBottom: 8 }}
                    />
                    <div style={{ fontSize: "0.78rem", color: "var(--on-surface-dim)", marginBottom: 4 }}>
                      {p.ocr_text?.slice(0, 30) || `Product ${i + 1}`}
                    </div>
                    <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)" }}>
                      Count: {p.count} | Conf: {(p.confidence * 100).toFixed(0)}%
                    </div>
                    <button
                      className="btn-primary"
                      style={{ marginTop: 8, padding: "6px 12px", fontSize: "0.75rem", width: "100%" }}
                      onClick={() => {
                        const name = prompt("Enter product name:", p.ocr_text?.slice(0, 30) || `Product ${i + 1}`);
                        if (name) handleBulkRegister(p, name);
                      }}
                    >
                      Register
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Registration Form + Catalog */}
        <div>
          {/* Registration Form */}
          <div className="surface-card" style={{ padding: 24, marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Registration</h3>
              <button
                className={`voice-btn ${isRecording ? "recording" : ""}`}
                onClick={toggleVoice}
                title={isRecording ? "Stop recording" : "Voice input"}
                style={{ width: 48, height: 48, fontSize: "1.2rem" }}
              >
                {isRecording ? "🔴" : "🎤"}
              </button>
            </div>

            {isRecording && (
              <div className="badge badge-danger" style={{ marginBottom: 12 }}>
                <span className="pulse-dot pulse-dot-red" /> Listening...
              </div>
            )}

            {/* Voice Guide */}
            <div style={{ padding: "12px 16px", background: "var(--surface-container)", borderRadius: "var(--radius-md)", marginBottom: 16, fontSize: "0.78rem", lineHeight: 1.6 }}>
              <div style={{ fontWeight: 700, marginBottom: 6, color: "var(--primary-bright)" }}>🎤 Voice Guide — Say something like:</div>
              <div style={{ color: "var(--on-surface-dim)" }}>
                <div>• <strong>&quot;Clean and Clear face wash price 120 rupees&quot;</strong></div>
                <div>• <strong>&quot;Parle-G biscuit Rs 10&quot;</strong></div>
                <div>• <strong>&quot;Amul milk 25 rupees&quot;</strong></div>
                <div>• <strong>&quot;Vim detergent price 45&quot;</strong></div>
                <div style={{ marginTop: 6, fontSize: "0.72rem", color: "var(--on-surface-muted)" }}>
                  💡 <em>Category auto-detects from keywords: biscuit→Snacks, milk→Dairy, shampoo→Personal Care, detergent→Household, masala→Spices, chocolate→Confectionery, juice→Beverages, rice/atta→Staples</em>
                </div>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label className="input-label">Product Name</label>
                <input className="input-field" placeholder="e.g. Parle-G Biscuits" value={productName} onChange={(e) => setProductName(e.target.value)} />
              </div>
              <div>
                <label className="input-label">SKU / Barcode</label>
                <input className="input-field" placeholder="Auto-detected or manual" value={productBarcode} onChange={(e) => setProductBarcode(e.target.value)} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label className="input-label">Category</label>
                  <select className="select-field" value={productCategory} onChange={(e) => setProductCategory(e.target.value)}>
                    {["Other", "Snacks", "Beverages", "Dairy", "Personal Care", "Household", "Staples", "Confectionery", "Spices"].map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="input-label">Price ₹</label>
                  <input className="input-field" type="number" placeholder="0.00" value={productPrice} onChange={(e) => setProductPrice(e.target.value)} />
                </div>
              </div>
              <button className="btn-primary" style={{ width: "100%", justifyContent: "center", padding: "14px 28px", fontSize: "0.95rem" }} onClick={handleRegister} disabled={registering}>
                {registering ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Registering...</> : "✅ Register Product"}
              </button>
            </div>
          </div>

          {/* Product Catalog */}
          <div className="surface-card" style={{ padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Product Catalog</h3>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="badge badge-info">{products.length} SKUs</span>
                {products.length > 0 && (
                  <button
                    style={{ background: "none", border: "none", cursor: "pointer", color: "var(--danger)", fontSize: "0.75rem", fontWeight: 600 }}
                    onClick={async () => {
                      if (!confirm("Delete ALL products? This cannot be undone.")) return;
                      try {
                        await apiFetch(`${API.products}`, { method: "DELETE" });
                        loadProducts();
                      } catch { /* ignore */ }
                    }}
                  >🗑️ Clear All</button>
                )}
              </div>
            </div>

            {products.length === 0 ? (
              <div style={{ textAlign: "center", padding: 32, color: "var(--on-surface-muted)" }}>
                <div style={{ fontSize: "2rem", marginBottom: 8 }}>📦</div>
                <div>No products registered yet.</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 400, overflowY: "auto" }}>
                {products.map((p) => (
                  <div
                    key={p.sku}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      padding: "10px 12px",
                      borderRadius: "var(--radius-md)",
                      background: "var(--surface-container)",
                    }}
                  >
                    {p.image_url ? (
                      <img src={`${API.base}${p.image_url}`} alt={p.name} style={{ width: 40, height: 40, borderRadius: 8, objectFit: "cover" }} />
                    ) : (
                      <div style={{ width: 40, height: 40, borderRadius: 8, background: "var(--surface-container-highest)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.1rem" }}>📦</div>
                    )}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: "0.85rem", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--on-surface-muted)" }}>{p.sku} • {p.category}</div>
                    </div>
                    {p.has_embedding && <span className="badge badge-success" style={{ fontSize: "0.65rem", padding: "2px 8px" }}>AI</span>}
                    <button
                      style={{ background: "none", border: "none", cursor: "pointer", color: "var(--danger)", fontSize: "0.85rem" }}
                      onClick={() => handleDelete(p.sku)}
                    >🗑️</button>
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
