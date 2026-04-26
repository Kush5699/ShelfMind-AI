"use client";

import { createContext, useContext, useState, ReactNode, useCallback, useRef } from "react";
import { API, apiFetch, apiUpload } from "./api";

/**
 * ShelfMind Global Store
 * Persists state AND background operations across page navigations.
 * Scan/monitor operations continue even when the user switches tabs.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface Product { sku: string; name: string; category: string; price: number; barcode?: string; image_url?: string; [key: string]: any; }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface ScanResult { [key: string]: any; }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface BulkResult { [key: string]: any; }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface MonitorResult { [key: string]: any; }

interface StoreContextType {
  // Products (shared across Scanner + other pages)
  products: Product[];
  loadProducts: () => Promise<void>;
  productsLoaded: boolean;

  // Scanner state
  scanResult: ScanResult | null;
  setScanResult: (r: ScanResult | null) => void;
  bulkResult: BulkResult | null;
  setBulkResult: (r: BulkResult | null) => void;
  scannerPreview: string | null;
  setScannerPreview: (img: string | null) => void;
  scannerFormState: { name: string; barcode: string; category: string; price: string };
  setScannerFormState: (s: { name: string; barcode: string; category: string; price: string }) => void;

  // Background scan operation (survives tab switches)
  scanning: boolean;
  scanStatus: string;
  scanError: string | null;
  startScan: (formData: FormData, mode: "single" | "bulk") => void;

  // Monitor state
  monitorResult: MonitorResult | null;
  setMonitorResult: (r: MonitorResult | null) => void;
  monitorPreview: string | null;
  setMonitorPreview: (img: string | null) => void;
  selectedPlanogram: string;
  setSelectedPlanogram: (p: string) => void;
}

const StoreContext = createContext<StoreContextType | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  // Products
  const [products, setProducts] = useState<Product[]>([]);
  const [productsLoaded, setProductsLoaded] = useState(false);

  const loadProducts = useCallback(async () => {
    try {
      const data = await apiFetch(API.products);
      setProducts(data.products || []);
      setProductsLoaded(true);
    } catch {
      /* ignore */
    }
  }, []);

  // Scanner
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [bulkResult, setBulkResult] = useState<BulkResult | null>(null);
  const [scannerPreview, setScannerPreview] = useState<string | null>(null);
  const [scannerFormState, setScannerFormState] = useState({
    name: "", barcode: "", category: "Other", price: "0.00",
  });

  // Background scan state (persists across tab switches)
  const [scanning, setScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState("");
  const [scanError, setScanError] = useState<string | null>(null);
  const scanAbortRef = useRef<AbortController | null>(null);

  const startScan = useCallback((formData: FormData, mode: "single" | "bulk") => {
    // Cancel any existing scan
    if (scanAbortRef.current) scanAbortRef.current.abort();
    const controller = new AbortController();
    scanAbortRef.current = controller;

    setScanning(true);
    setScanResult(null);
    setBulkResult(null);
    setScanError(null);
    setScanStatus(mode === "single"
      ? "Running AI pipeline (rembg → DINOv2 10-view embedding)..."
      : "Detecting products on shelf...");

    // Run the scan in the store context — survives page navigation
    (async () => {
      try {
        if (mode === "single") {
          const result = await apiUpload(API.scanSingle, formData);
          if (!controller.signal.aborted) setScanResult(result);
        } else {
          const result = await apiUpload(API.scanBulk, formData);
          if (!controller.signal.aborted) setBulkResult(result);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setScanError((err as Error).message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setScanning(false);
          setScanStatus("");
        }
      }
    })();
  }, []);

  // Monitor
  const [monitorResult, setMonitorResult] = useState<MonitorResult | null>(null);
  const [monitorPreview, setMonitorPreview] = useState<string | null>(null);
  const [selectedPlanogram, setSelectedPlanogram] = useState("");

  return (
    <StoreContext.Provider
      value={{
        products, loadProducts, productsLoaded,
        scanResult, setScanResult,
        bulkResult, setBulkResult,
        scannerPreview, setScannerPreview,
        scannerFormState, setScannerFormState,
        scanning, scanStatus, scanError, startScan,
        monitorResult, setMonitorResult,
        monitorPreview, setMonitorPreview,
        selectedPlanogram, setSelectedPlanogram,
      }}
    >
      {children}
    </StoreContext.Provider>
  );
}

export function useStore() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore must be used within StoreProvider");
  return ctx;
}
