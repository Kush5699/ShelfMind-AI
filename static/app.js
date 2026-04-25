/**
 * ============================================================================
 * ShelfMind AI — Frontend Application Logic
 * ============================================================================
 * Handles navigation, API calls, voice registration, camera, live monitoring.
 * ============================================================================
 */

const API = window.location.origin;

// ── State ────────────────────────────────────────────────
const state = {
    scans: 0,
    totalProducts: 0,
    complianceScores: [],
    latencies: [],
    currentImage: null,
    liveStream: null,
    liveInterval: null,
    liveRunning: false,
    liveFrameCount: 0,
    liveTotalProducts: 0,
    liveLatencies: [],
    ws: null,
    voiceRecognition: null,
    voiceActive: false,
};

// ── Navigation ───────────────────────────────────────────
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`page-${tab.dataset.page}`).classList.add('active');

        if (tab.dataset.page === 'catalog') loadProducts();
    });
});

// ── Toast Notifications ──────────────────────────────────
function showToast(msg, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `${icons[type] || ''} ${msg}`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 4000);
}

// ── System Health Check ──────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();
        document.getElementById('statusText').textContent =
            data.status === 'healthy' ? `Online • ${data.device.toUpperCase()}` : 'Loading...';
        document.getElementById('systemStatus').style.borderColor =
            data.status === 'healthy' ? 'rgba(0,212,170,0.3)' : 'rgba(255,107,107,0.3)';

        if (data.models?.faiss_vectors) {
            document.getElementById('metricProducts').textContent = data.models.faiss_vectors.toLocaleString();
        }
    } catch (e) {
        document.getElementById('statusText').textContent = 'Offline';
        document.getElementById('systemStatus').style.borderColor = 'rgba(255,107,107,0.3)';
    }
}

checkHealth();
setInterval(checkHealth, 30000);

// ═══════════════════════════════════════════════════════════
// SHELF ANALYZER
// ═══════════════════════════════════════════════════════════

const analyzerDropZone = document.getElementById('analyzerDropZone');
const analyzerFileInput = document.getElementById('analyzerFileInput');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const btnAnalyze = document.getElementById('btnAnalyze');

// Confidence slider
confSlider.addEventListener('input', () => {
    confValue.textContent = (confSlider.value / 100).toFixed(2);
});

// Drag & Drop
analyzerDropZone.addEventListener('click', () => analyzerFileInput.click());
analyzerDropZone.addEventListener('dragover', e => { e.preventDefault(); analyzerDropZone.classList.add('dragover'); });
analyzerDropZone.addEventListener('dragleave', () => analyzerDropZone.classList.remove('dragover'));
analyzerDropZone.addEventListener('drop', e => {
    e.preventDefault();
    analyzerDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleAnalyzerFile(e.dataTransfer.files[0]);
});

analyzerFileInput.addEventListener('change', e => {
    if (e.target.files.length) handleAnalyzerFile(e.target.files[0]);
});

// Camera button
document.getElementById('btnCamera').addEventListener('click', () => {
    analyzerFileInput.setAttribute('capture', 'environment');
    analyzerFileInput.click();
});

function handleAnalyzerFile(file) {
    if (!file.type.startsWith('image/')) { showToast('Please select an image', 'error'); return; }
    state.currentImage = file;
    btnAnalyze.disabled = false;

    // Preview
    const reader = new FileReader();
    reader.onload = e => {
        analyzerDropZone.innerHTML = `
            <img src="${e.target.result}" style="max-height:300px; border-radius:12px; margin:0 auto;">
            <div class="upload-hint" style="margin-top:12px;">Click "Analyze Shelf" to process</div>
        `;
    };
    reader.readAsDataURL(file);
    showToast('Image loaded — click Analyze', 'success');
}

// Analyze button
btnAnalyze.addEventListener('click', analyzeShelf);

async function analyzeShelf() {
    if (!state.currentImage) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = '<div class="spinner" style="width:16px;height:16px;margin:0;border-width:2px;"></div> Analyzing...';

    const formData = new FormData();
    formData.append('image', state.currentImage);
    formData.append('confidence', (confSlider.value / 100).toFixed(2));
    formData.append('annotate', 'true');

    try {
        const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: formData });
        const data = await res.json();

        // Update state
        state.scans++;
        state.latencies.push(data.inference_ms);
        if (data.compliance_score !== undefined) {
            state.complianceScores.push(data.compliance_score);
        }

        // Show results
        document.getElementById('analyzerUploadCard').style.display = 'none';
        document.getElementById('analyzerResultCard').style.display = 'block';
        document.getElementById('analyzerStatsCard').style.display = 'block';

        if (data.annotated_image) {
            document.getElementById('analyzerResultImg').src = `data:image/jpeg;base64,${data.annotated_image}`;
        }
        document.getElementById('analyzerOverlay').textContent = `${data.total_products} products detected`;

        // Gauge
        const score = data.compliance_score || 0;
        const circumference = 440;
        const offset = circumference - (circumference * score / 100);
        document.getElementById('gaugeCircle').style.strokeDashoffset = offset;
        document.getElementById('complianceScore').textContent = `${score}%`;

        // Details
        document.getElementById('analysisDetails').innerHTML = `
            <div>🔍 Products Detected: <strong>${data.total_products}</strong></div>
            <div>✅ Products Identified: <strong>${data.identified_products}</strong></div>
            <div>⏱️ Inference Time: <strong>${data.inference_ms} ms</strong></div>
            <div>📐 Image Size: <strong>${data.image_size.width}×${data.image_size.height}</strong></div>
        `;

        // Update dashboard metrics
        updateDashboardMetrics();

        // Add to activity
        addActivity(`Shelf scan: ${data.total_products} products, ${score}% compliance (${data.inference_ms}ms)`);

        showToast(`Detected ${data.total_products} products!`, 'success');
    } catch (e) {
        showToast(`Analysis failed: ${e.message}`, 'error');
    }

    btnAnalyze.disabled = false;
    btnAnalyze.innerHTML = '🔍 Analyze Shelf';
}

function resetAnalyzer() {
    document.getElementById('analyzerUploadCard').style.display = 'block';
    document.getElementById('analyzerResultCard').style.display = 'none';
    document.getElementById('analyzerStatsCard').style.display = 'none';
    state.currentImage = null;
    btnAnalyze.disabled = true;

    analyzerDropZone.innerHTML = `
        <div class="upload-icon">📸</div>
        <div class="upload-text">Drop shelf photo here or click to upload</div>
        <div class="upload-hint">Supports JPG, PNG • Max 20 MB</div>
    `;
}

function updateDashboardMetrics() {
    document.getElementById('metricScans').textContent = state.scans;
    if (state.complianceScores.length) {
        const avg = state.complianceScores.reduce((a, b) => a + b, 0) / state.complianceScores.length;
        document.getElementById('metricCompliance').textContent = `${avg.toFixed(1)}%`;
    }
    if (state.latencies.length) {
        const avg = state.latencies.reduce((a, b) => a + b, 0) / state.latencies.length;
        document.getElementById('metricLatency').textContent = `${avg.toFixed(0)}ms`;
    }
}

function addActivity(text) {
    const container = document.getElementById('recentActivity');
    if (container.querySelector('.empty-state')) container.innerHTML = '';

    const time = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.style.cssText = 'padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.85rem;';
    div.innerHTML = `<span style="color:var(--text-muted); font-size:0.7rem;">${time}</span> — ${text}`;
    container.prepend(div);
}

// ═══════════════════════════════════════════════════════════
// PRODUCT CATALOG
// ═══════════════════════════════════════════════════════════

function openAddProduct() {
    document.getElementById('addProductForm').style.display = 'block';
}

function closeAddProduct() {
    document.getElementById('addProductForm').style.display = 'none';
    document.getElementById('inputName').value = '';
    document.getElementById('inputPrice').value = '';
    document.getElementById('inputSKU').value = '';
}

async function saveProduct() {
    const name = document.getElementById('inputName').value.trim();
    if (!name) { showToast('Product name required!', 'error'); return; }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('category', document.getElementById('inputCategory').value);
    formData.append('price', document.getElementById('inputPrice').value || '0');
    formData.append('sku', document.getElementById('inputSKU').value);

    const imgInput = document.getElementById('inputImage');
    if (imgInput.files.length) formData.append('image', imgInput.files[0]);

    try {
        const res = await fetch(`${API}/api/products`, { method: 'POST', body: formData });
        const data = await res.json();
        showToast(`"${name}" added successfully!`, 'success');
        closeAddProduct();
        loadProducts();
    } catch (e) {
        showToast(`Failed to add product: ${e.message}`, 'error');
    }
}

async function loadProducts() {
    try {
        const res = await fetch(`${API}/api/products`);
        const data = await res.json();
        const products = data.products || [];
        const container = document.getElementById('productList');

        if (!products.length) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon">📦</div><div>No products registered yet</div></div>';
            return;
        }

        container.innerHTML = `<div class="product-grid">${products.map(p => `
            <div class="product-card">
                <div class="product-name">${p.name || 'Unknown'}</div>
                <div class="product-sku">${p.sku || '—'}</div>
                <div class="product-category">${p.category || 'General'}</div>
                ${p.price ? `<div class="product-price">₹${p.price}</div>` : ''}
                <button class="btn btn-ghost" style="margin-top:8px; font-size:0.75rem; color:var(--accent-4);"
                        onclick="deleteProduct(${p.id})">🗑 Remove</button>
            </div>
        `).join('')}</div>`;

        document.getElementById('metricProducts').textContent = products.length;
    } catch (e) {
        // Silent fail
    }
}

async function deleteProduct(id) {
    try {
        await fetch(`${API}/api/products/${id}`, { method: 'DELETE' });
        showToast('Product removed', 'info');
        loadProducts();
    } catch (e) {
        showToast('Failed to delete', 'error');
    }
}

// ═══════════════════════════════════════════════════════════
// VOICE REGISTRATION
// ═══════════════════════════════════════════════════════════

function toggleVoice() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast('Voice recognition not supported in this browser. Use Chrome!', 'error');
        return;
    }

    const voiceBtn = document.getElementById('btnVoice');
    const voiceStatus = document.getElementById('voiceStatus');

    if (state.voiceActive) {
        state.voiceRecognition.stop();
        state.voiceActive = false;
        voiceBtn.classList.remove('recording');
        voiceStatus.textContent = 'Click to speak';
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
        state.voiceActive = true;
        voiceBtn.classList.add('recording');
        voiceStatus.textContent = '🔴 Listening...';
    };

    recognition.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        voiceStatus.textContent = `"${transcript}"`;

        if (event.results[0].isFinal) {
            voiceBtn.classList.remove('recording');
            state.voiceActive = false;

            // Parse with backend
            try {
                const formData = new FormData();
                formData.append('transcript', transcript);
                const res = await fetch(`${API}/api/products/voice`, { method: 'POST', body: formData });
                const data = await res.json();

                if (data.parsed) {
                    document.getElementById('inputName').value = data.parsed.name;
                    document.getElementById('inputPrice').value = data.parsed.price || '';
                    document.getElementById('inputSKU').value = data.parsed.sku || '';

                    // Set category
                    const catSelect = document.getElementById('inputCategory');
                    for (let opt of catSelect.options) {
                        if (opt.value === data.parsed.category) { catSelect.value = opt.value; break; }
                    }

                    voiceStatus.textContent = `✅ Parsed: ${data.parsed.name}`;
                    showToast(`Voice parsed: "${data.parsed.name}" (${data.parsed.category}, ₹${data.parsed.price})`, 'success');
                }
            } catch (e) {
                voiceStatus.textContent = 'Parse failed';
                // Fallback: just set name
                document.getElementById('inputName').value = transcript;
            }
        }
    };

    recognition.onerror = (e) => {
        voiceBtn.classList.remove('recording');
        state.voiceActive = false;
        voiceStatus.textContent = `Error: ${e.error}`;
        showToast(`Voice error: ${e.error}`, 'error');
    };

    recognition.onend = () => {
        state.voiceActive = false;
        voiceBtn.classList.remove('recording');
    };

    state.voiceRecognition = recognition;
    recognition.start();
}

// ═══════════════════════════════════════════════════════════
// LIVE MONITORING
// ═══════════════════════════════════════════════════════════

const liveFPSSlider = document.getElementById('liveFPSSlider');
const liveFPSValue = document.getElementById('liveFPSValue');
liveFPSSlider.addEventListener('input', () => { liveFPSValue.textContent = liveFPSSlider.value; });

async function startLive() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });

        state.liveStream = stream;
        const video = document.getElementById('liveVideo');
        video.srcObject = stream;
        video.style.display = 'block';
        video.play();

        document.getElementById('liveIdle').style.display = 'none';
        document.getElementById('liveBadge').style.display = 'flex';
        document.getElementById('liveStats').style.display = 'flex';
        document.getElementById('btnStartLive').disabled = true;
        document.getElementById('btnStopLive').disabled = false;

        state.liveRunning = true;
        state.liveFrameCount = 0;
        state.liveTotalProducts = 0;
        state.liveLatencies = [];

        // Start sending frames
        const fps = parseInt(liveFPSSlider.value);
        state.liveInterval = setInterval(() => captureAndDetect(), 1000 / fps);

        showToast('Live monitoring started!', 'success');
    } catch (e) {
        showToast(`Camera error: ${e.message}`, 'error');
    }
}

function stopLive() {
    state.liveRunning = false;
    if (state.liveInterval) clearInterval(state.liveInterval);
    if (state.liveStream) {
        state.liveStream.getTracks().forEach(t => t.stop());
    }

    document.getElementById('liveVideo').style.display = 'none';
    document.getElementById('liveIdle').style.display = 'flex';
    document.getElementById('liveBadge').style.display = 'none';
    document.getElementById('liveStats').style.display = 'none';
    document.getElementById('btnStartLive').disabled = false;
    document.getElementById('btnStopLive').disabled = true;

    showToast('Live monitoring stopped', 'info');
}

async function captureAndDetect() {
    if (!state.liveRunning) return;

    const video = document.getElementById('liveVideo');
    const canvas = document.getElementById('liveCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob(async (blob) => {
        if (!blob || !state.liveRunning) return;

        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        formData.append('confidence', '0.3');
        formData.append('annotate', 'false');

        try {
            const start = performance.now();
            const res = await fetch(`${API}/api/detect`, { method: 'POST', body: formData });
            const data = await res.json();
            const latency = Math.round(performance.now() - start);

            state.liveFrameCount++;
            state.liveTotalProducts += data.total_products;
            state.liveLatencies.push(latency);

            // Draw detections on canvas
            ctx.drawImage(video, 0, 0);
            const colors = ['#00d4aa', '#00b4d8', '#7b68ee', '#ff6b6b', '#ffe66d'];
            (data.detections || []).forEach((det, i) => {
                const [x1, y1, x2, y2] = det.bbox;
                ctx.strokeStyle = colors[i % colors.length];
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.fillStyle = colors[i % colors.length];
                ctx.fillRect(x1, y1 - 18, 50, 18);
                ctx.fillStyle = '#000';
                ctx.font = '12px Inter, sans-serif';
                ctx.fillText(`${(det.confidence * 100).toFixed(0)}%`, x1 + 4, y1 - 4);
            });

            // Update stats
            document.getElementById('liveProductCount').textContent = `Products: ${data.total_products}`;
            document.getElementById('liveLatency').textContent = `Latency: ${latency}ms`;
            document.getElementById('liveFrameCount').textContent = state.liveFrameCount;
            document.getElementById('liveTotalProducts').textContent = state.liveTotalProducts;
            document.getElementById('liveAvgProducts').textContent =
                (state.liveTotalProducts / state.liveFrameCount).toFixed(1);
            const avgLat = state.liveLatencies.reduce((a, b) => a + b, 0) / state.liveLatencies.length;
            document.getElementById('liveAvgLatency').textContent = `${avgLat.toFixed(0)}ms`;
            document.getElementById('liveFPS').textContent =
                `FPS: ${(1000 / avgLat).toFixed(1)}`;

        } catch (e) {
            // Silently skip failed frames
        }
    }, 'image/jpeg', 0.7);
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════

loadProducts();
