/**
 * Real-Time Virtual Clothing Try-On Web Engine (60 FPS GPU Compositing + Async 80x80 ONNX Mask Decode)
 * Tách riêng luồng render camera (GPU drawImage) khỏi luồng suy luận ONNX để giữ camera 30-60 FPS mượt mà trên iPhone Safari.
 */

const INPUT_SIZE = 320;
const CONF_THRESHOLD = 0.45;
const NUM_MASK_COEFFS = 32;
const BBOX_EMA_ALPHA = 0.45;

export const GARMENT_CATEGORIES = [
  { id: 0, key: "shirt", label: "Áo thun" },
  { id: 1, key: "pants", label: "Quần" },
  { id: 2, key: "jacket", label: "Áo khoác" },
  { id: 3, key: "dress", label: "Váy" },
];

export class TryOnWebEngine {
  constructor() {
    this.session = null;
    this.inputName = "images";
    this.providerName = "Smart Preview";

    // Buffer 320x320 tái sử dụng cho suy luận AI (không cấp phát mỗi frame)
    this.inputFloatBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    this.aiCanvas = document.createElement("canvas");
    this.aiCanvas.width = INPUT_SIZE;
    this.aiCanvas.height = INPUT_SIZE;
    this.aiCtx = this.aiCanvas.getContext("2d", { willReadFrequently: true });

    // Canvas 80x80 giải mã mask trực tiếp ở độ phân giải prototype + GPU compositing
    this.maskCanvas = document.createElement("canvas");
    this.maskCtx = this.maskCanvas.getContext("2d");
    this.maskImageData = null;

    // Canvas lưu trang phục đã cắt sẵn alpha theo mask để render 60 FPS lên camera
    this.garmentSrcCanvas = document.createElement("canvas");
    this.garmentSrcCtx = this.garmentSrcCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    this.overlayCanvas = document.createElement("canvas");
    this.overlayCtx = this.overlayCanvas.getContext("2d");

    this.hasGarment = false;
    this.isInferring = false;
    this.lastAiMs = 0;

    // Kết quả detection mới nhất (đã làm mượt EMA) để luồng vẽ 60 FPS dùng ngay
    this.latestOverlay = null; // { bbox: [x1,y1,x2,y2], confidence, updatedAt }
    this.smoothedBbox = null;
    this.lastTargetClass = null;
  }

  get hasModel() {
    return this.session !== null;
  }

  async tryLoadRustWasm() {
    // Giữ hook tương thích cho build wasm-pack tùy chọn
    try {
      const mod = await import("./rust_wasm/pkg/rust_tryon_wasm.js");
      if (typeof mod.default === "function") {
        await mod.default();
      }
    } catch {
      // Sử dụng đường chạy WebGL/WASM + GPU Canvas mặc định
    }
  }

  async loadModel(source) {
    const ort = window.ort;
    if (!ort) {
      throw new Error("Không tải được thư viện onnxruntime-web.");
    }

    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;

    let modelBytes = source;
    if (typeof source === "string") {
      const resp = await fetch(source, { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`Không tìm thấy file model (${resp.status})`);
      }
      modelBytes = await resp.arrayBuffer();
    }

    // Ưu tiên WebGPU -> WebGL -> WASM SIMD trên Safari/Chrome di động
    const providersToTry = [
      { name: "WebGPU", opts: { executionProviders: ["webgpu", "wasm"] } },
      { name: "WebGL", opts: { executionProviders: ["webgl", "wasm"] } },
      { name: "WASM", opts: { executionProviders: ["wasm"] } },
    ];

    let lastErr = null;
    for (const candidate of providersToTry) {
      try {
        const session = await ort.InferenceSession.create(modelBytes, {
          ...candidate.opts,
          graphOptimizationLevel: "all",
        });
        this.session = session;
        this.inputName = session.inputNames?.[0] || "images";
        this.providerName = `ONNX ${candidate.name}`;
        return this.providerName;
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error("Không thể khởi tạo ONNX InferenceSession.");
  }

  setGarmentFromSource(imgOrCanvas, autoRemoveWhiteBg = false) {
    const w = imgOrCanvas.naturalWidth || imgOrCanvas.width;
    const h = imgOrCanvas.naturalHeight || imgOrCanvas.height;
    if (!w || !h) return;

    this.garmentSrcCanvas.width = w;
    this.garmentSrcCanvas.height = h;
    this.garmentSrcCtx.clearRect(0, 0, w, h);
    this.garmentSrcCtx.drawImage(imgOrCanvas, 0, 0, w, h);

    if (autoRemoveWhiteBg) {
      const imgData = this.garmentSrcCtx.getImageData(0, 0, w, h);
      removeStudioBackgroundInPlace(imgData.data, w, h);
      this.garmentSrcCtx.putImageData(imgData, 0, 0);
    }
    this.hasGarment = true;
  }

  setSampleGarment(categoryId, colorHex) {
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = colorHex;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 5;
    ctx.beginPath();

    if (categoryId === 0 || categoryId === 2) {
      ctx.moveTo(64, 24);
      ctx.lineTo(100, 16);
      ctx.quadraticCurveTo(128, 44, 156, 16);
      ctx.lineTo(192, 24);
      ctx.lineTo(244, 88);
      ctx.lineTo(208, 118);
      ctx.lineTo(192, 92);
      ctx.lineTo(192, 240);
      ctx.lineTo(64, 240);
      ctx.lineTo(64, 92);
      ctx.lineTo(48, 118);
      ctx.lineTo(12, 88);
      ctx.closePath();
    } else if (categoryId === 1) {
      ctx.moveTo(56, 16);
      ctx.lineTo(200, 16);
      ctx.lineTo(212, 244);
      ctx.lineTo(148, 244);
      ctx.lineTo(128, 108);
      ctx.lineTo(108, 244);
      ctx.lineTo(44, 244);
      ctx.closePath();
    } else {
      ctx.moveTo(84, 16);
      ctx.quadraticCurveTo(128, 42, 172, 16);
      ctx.lineTo(180, 86);
      ctx.lineTo(232, 244);
      ctx.lineTo(24, 244);
      ctx.lineTo(76, 86);
      ctx.closePath();
    }

    ctx.fill();
    ctx.stroke();
    this.setGarmentFromSource(canvas, false);
  }

  /**
   * Luồng vẽ 60 FPS (chỉ dùng GPU drawImage, KHÔNG đụng getImageData hay await ONNX).
   */
  renderCameraAndOverlay(video, outCtx, width, height, mirror, targetClassId) {
    outCtx.save();
    if (mirror) {
      outCtx.translate(width, 0);
      outCtx.scale(-1, 1);
    }
    outCtx.drawImage(video, 0, 0, width, height);
    outCtx.restore();

    if (!this.hasGarment) return;

    if (!this.session) {
      // Smart Preview trực tiếp bằng GPU
      const box = smartPreviewBox(targetClassId);
      outCtx.save();
      outCtx.globalAlpha = 0.88;
      outCtx.drawImage(
        this.garmentSrcCanvas,
        box[0] * width,
        box[1] * height,
        (box[2] - box[0]) * width,
        (box[3] - box[1]) * height
      );
      outCtx.restore();
      return;
    }

    const overlay = this.latestOverlay;
    if (!overlay || performance.now() - overlay.updatedAt > 600) {
      return;
    }

    const [x1, y1, x2, y2] = overlay.bbox;
    const dx = x1 * width;
    const dy = y1 * height;
    const dw = Math.max(1, (x2 - x1) * width);
    const dh = Math.max(1, (y2 - y1) * height);

    outCtx.drawImage(this.overlayCanvas, dx, dy, dw, dh);
  }

  /**
   * Luồng suy luận AI bất đồng bộ: thu nhỏ trên GPU 320x320 + giải mã mask 80x80 + cắt áo bằng GPU destination-in.
   */
  async runInferenceStep(video, mirror, targetClassId) {
    if (!this.session || !this.hasGarment || this.isInferring) return;
    if (video.readyState < 2 || video.videoWidth <= 0) return;

    this.isInferring = true;
    const t0 = performance.now();

    try {
      if (this.lastTargetClass !== targetClassId) {
        this.smoothedBbox = null;
        this.latestOverlay = null;
        this.lastTargetClass = targetClassId;
      }

      // GPU thu nhỏ video về 320x320 (tốn < 0.3 ms)
      this.aiCtx.save();
      if (mirror) {
        this.aiCtx.translate(INPUT_SIZE, 0);
        this.aiCtx.scale(-1, 1);
      }
      this.aiCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
      this.aiCtx.restore();

      const imgData = this.aiCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
      const planeSize = INPUT_SIZE * INPUT_SIZE;
      const out = this.inputFloatBuffer;
      const inv255 = 1.0 / 255.0;

      for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
        out[i] = imgData[p] * inv255;
        out[planeSize + i] = imgData[p + 1] * inv255;
        out[2 * planeSize + i] = imgData[p + 2] * inv255;
      }

      const ort = window.ort;
      const inputTensor = new ort.Tensor("float32", out, [
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
      ]);

      const results = await this.session.run({ [this.inputName]: inputTensor });
      const keys = Object.keys(results);
      if (keys.length < 2) return;

      const best = findBestDetection(results[keys[0]], targetClassId);
      if (!best) {
        this.smoothedBbox = null;
        this.latestOverlay = null;
        return;
      }

      const bbox = smoothBbox(this.smoothedBbox, best.bbox);
      this.smoothedBbox = bbox;

      const protosTensor = results[keys[1]];
      const pDims = protosTensor.dims;
      if (!pDims || pDims.length < 4 || pDims[1] < NUM_MASK_COEFFS) return;

      const mh = pDims[2];
      const mw = pDims[3];
      this._buildCutoutOverlay(protosTensor.data, mh, mw, best.coeffs, bbox);

      this.latestOverlay = {
        bbox,
        confidence: best.confidence,
        updatedAt: performance.now(),
      };
    } catch (err) {
      console.error("ONNX inference error:", err);
    } finally {
      this.lastAiMs = Math.round(performance.now() - t0);
      this.isInferring = false;
    }
  }

  _buildCutoutOverlay(protos, mh, mw, coeffs, bbox) {
    if (this.maskCanvas.width !== mw || this.maskCanvas.height !== mh) {
      this.maskCanvas.width = mw;
      this.maskCanvas.height = mh;
      this.maskImageData = this.maskCtx.createImageData(mw, mh);
    }

    const maskPixels = this.maskImageData.data;
    maskPixels.fill(0);

    const px1 = Math.max(0, Math.floor(bbox[0] * mw) - 1);
    const py1 = Math.max(0, Math.floor(bbox[1] * mh) - 1);
    const px2 = Math.min(mw, Math.ceil(bbox[2] * mw) + 1);
    const py2 = Math.min(mh, Math.ceil(bbox[3] * mh) + 1);
    const planeSize = mh * mw;

    // Giải mã mask chỉ trên lưới 80x80 trong phạm vi bounding box
    for (let py = py1; py < py2; py++) {
      const rowOff = py * mw;
      for (let px = px1; px < px2; px++) {
        const pOff = rowOff + px;
        let sum = 0.0;
        for (let c = 0; c < NUM_MASK_COEFFS; c++) {
          sum += coeffs[c] * protos[c * planeSize + pOff];
        }
        const mVal = 1.0 / (1.0 + Math.exp(-sum));
        if (mVal > 0.42) {
          const alpha = mVal >= 0.56 ? 255 : Math.round(((mVal - 0.42) / 0.14) * 255);
          maskPixels[pOff * 4 + 3] = alpha;
        }
      }
    }
    this.maskCtx.putImageData(this.maskImageData, 0, 0);

    // Tạo overlay kích thước vừa đủ (tối đa 320px) và cắt theo mask bằng GPU
    const ow = Math.max(32, Math.round((bbox[2] - bbox[0]) * INPUT_SIZE));
    const oh = Math.max(32, Math.round((bbox[3] - bbox[1]) * INPUT_SIZE));
    if (this.overlayCanvas.width !== ow || this.overlayCanvas.height !== oh) {
      this.overlayCanvas.width = ow;
      this.overlayCanvas.height = oh;
    }

    const oCtx = this.overlayCtx;
    oCtx.globalCompositeOperation = "source-over";
    oCtx.clearRect(0, 0, ow, oh);
    oCtx.drawImage(this.garmentSrcCanvas, 0, 0, ow, oh);

    // Cắt mask theo đúng phần bbox trên lưới 80x80
    oCtx.globalCompositeOperation = "destination-in";
    oCtx.drawImage(
      this.maskCanvas,
      bbox[0] * mw,
      bbox[1] * mh,
      Math.max(1, (bbox[2] - bbox[0]) * mw),
      Math.max(1, (bbox[3] - bbox[1]) * mh),
      0,
      0,
      ow,
      oh
    );
    oCtx.globalCompositeOperation = "source-over";
  }
}

function smartPreviewBox(targetClassId) {
  if (targetClassId === 1) return [0.28, 0.48, 0.72, 0.94];
  if (targetClassId === 3) return [0.24, 0.26, 0.76, 0.88];
  return [0.22, 0.24, 0.78, 0.68];
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function smoothBbox(prev, curr) {
  if (!prev) return curr;
  const a = BBOX_EMA_ALPHA;
  const b = 1.0 - a;
  return [
    a * curr[0] + b * prev[0],
    a * curr[1] + b * prev[1],
    a * curr[2] + b * prev[2],
    a * curr[3] + b * prev[3],
  ];
}

function findBestDetection(detsTensor, targetClass) {
  const dims = detsTensor.dims;
  if (!dims || dims.length < 3) return null;

  const channels = dims[1];
  const anchors = dims[2];
  if (channels <= 4 + NUM_MASK_COEFFS) return null;

  const numClasses = channels - 4 - NUM_MASK_COEFFS;
  if (targetClass < 0 || targetClass >= numClasses) return null;

  const data = detsTensor.data;
  let bestConf = CONF_THRESHOLD;
  let bestIdx = -1;

  const confOffset = (4 + targetClass) * anchors;
  for (let i = 0; i < anchors; i++) {
    const conf = data[confOffset + i];
    if (conf > bestConf) {
      bestConf = conf;
      bestIdx = i;
    }
  }

  if (bestIdx === -1) return null;

  const invInput = 1.0 / INPUT_SIZE;
  const cx = data[bestIdx] * invInput;
  const cy = data[anchors + bestIdx] * invInput;
  const w = data[2 * anchors + bestIdx] * invInput;
  const h = data[3 * anchors + bestIdx] * invInput;
  const bbox = [
    clamp01(cx - w * 0.5),
    clamp01(cy - h * 0.5),
    clamp01(cx + w * 0.5),
    clamp01(cy + h * 0.5),
  ];

  const coeffs = new Float32Array(NUM_MASK_COEFFS);
  const coeffBase = (4 + numClasses) * anchors;
  for (let m = 0; m < NUM_MASK_COEFFS; m++) {
    coeffs[m] = data[coeffBase + m * anchors + bestIdx];
  }

  return { bbox, coeffs, confidence: bestConf };
}

function removeStudioBackgroundInPlace(data, width, height) {
  const corners = [
    0,
    (width - 1) * 4,
    (height - 1) * width * 4,
    ((height - 1) * width + (width - 1)) * 4,
  ];
  let bgR = 0,
    bgG = 0,
    bgB = 0;
  for (const idx of corners) {
    bgR += data[idx];
    bgG += data[idx + 1];
    bgB += data[idx + 2];
  }
  bgR /= 4;
  bgG /= 4;
  bgB /= 4;

  const cornerLuma = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;
  if (cornerLuma < 185) return;

  for (let i = 0; i < data.length; i += 4) {
    const dr = data[i] - bgR;
    const dg = data[i + 1] - bgG;
    const db = data[i + 2] - bgB;
    const dist = Math.sqrt(dr * dr + dg * dg + db * db);
    if (dist < 28) {
      data[i + 3] = 0;
    } else if (dist < 48) {
      data[i + 3] = Math.round(((dist - 28) / 20) * 255);
    }
  }
}
