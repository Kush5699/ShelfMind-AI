// ShelfMind AI — API Configuration
// Points to the HuggingFace Spaces backend

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://kush5699-shelfmind-ai.hf.space";

export const API = {
  base: API_BASE,
  health: `${API_BASE}/api/health`,
  // Scanner
  scanSingle: `${API_BASE}/api/scan/single`,
  scanBulk: `${API_BASE}/api/scan/bulk`,
  // Products
  products: `${API_BASE}/api/products`,
  productVoice: `${API_BASE}/api/products/voice`,
  productImage: (sku: string) => `${API_BASE}/api/products/${sku}/image`,
  productDelete: (sku: string) => `${API_BASE}/api/products/${sku}`,
  // Planograms
  planogramAutoDetect: `${API_BASE}/api/planogram/auto-detect`,
  planograms: `${API_BASE}/api/planograms`,
  planogramDelete: (name: string) => `${API_BASE}/api/planograms/${name}`,
  // Compliance
  complianceCheck: `${API_BASE}/api/compliance/check`,
  complianceHistory: `${API_BASE}/api/compliance/history`,
  // Analytics
  analyticsSummary: `${API_BASE}/api/analytics/summary`,
  alerts: `${API_BASE}/api/alerts`,
  // Notifications
  notify: `${API_BASE}/api/notify`,
  // WebSocket
  wsMonitor: API_BASE.replace("https://", "wss://").replace("http://", "ws://") + "/ws/monitor",
};

// Fetch helper with error handling
export async function apiFetch(url: string, options?: RequestInit) {
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options?.headers,
      },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API Error ${res.status}: ${text}`);
    }
    return res.json();
  } catch (err) {
    console.error(`[API] ${url}:`, err);
    throw err;
  }
}

// Upload helper (FormData)
export async function apiUpload(url: string, formData: FormData) {
  try {
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Upload Error ${res.status}: ${text}`);
    }
    return res.json();
  } catch (err) {
    console.error(`[API Upload] ${url}:`, err);
    throw err;
  }
}

/**
 * Compress an image blob/file by resizing to maxDim before upload.
 * Reduces 12MP phone images (4-5MB) to ~200KB, cutting upload + processing time dramatically.
 */
export function compressImage(
  input: Blob | File | string,
  maxDim = 1024,
  quality = 0.85
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      // Only downscale, never upscale
      if (width > maxDim || height > maxDim) {
        const scale = maxDim / Math.max(width, height);
        width = Math.round(width * scale);
        height = Math.round(height * scale);
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Canvas not supported"));
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Compression failed"))),
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => reject(new Error("Image load failed"));
    if (typeof input === "string") {
      img.src = input;
    } else {
      img.src = URL.createObjectURL(input);
    }
  });
}
