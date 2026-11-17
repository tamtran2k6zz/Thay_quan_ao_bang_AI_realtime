/**
 * Real-Time Virtual Clothing Try-On Web Engine (ONNX Runtime Web + TypedArray / optional Rust WASM)
 * Chuyển thể trực tiếp từ `src/lib.rs` để chạy mượt trên Safari iPhone & trình duyệt di động.
 */

const INPUT_SIZE = 320;
const CONF_THRESHOLD = 0.45;
const NUM_MASK_COEFFS = 32;

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
    this.providerName = "Smart Preview (chưa nạp .onnx)";
    this.inputFloatBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

    // Offscreen canvas dùng để resize ảnh trang phục mẫu bằng GPU trình duyệt
    this.garmentResizeCanvas = document.createElement("canvas");
    this.garmentResizeCtx = this.garmentResizeCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    this.garmentSrcCanvas = document.createElement("canvas");
    this.garmentSrcCtx = this.garmentSrcCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    this.garmentImageData = null;
    this.wasmDecodeAndBlend = null;
    this.isBusy = false;
  }

  get hasModel() {
    return this.session !== null;
  }

  /**
   * Tự động nạp module Rust WASM nếu người dùng đã build trong `web/rust_wasm/pkg/`.
   */
  async tryLoadRustWasm() {
    try {
      const mod = await import("./rust_wasm/pkg/rust_tryon_wasm.js");
      if (typeof mod.default === "function") {
        await mod.default();
      }
      if (typeof mod.decode_and_blend_wasm === "function") {
        this.wasmDecodeAndBlend = mod.decode_and_blend_wasm;
      }
    } catch {
      // Không bắt buộc: sử dụng TypedArray JS thuần mặc định
    }
  }

  /**
   * Nạp file model ONNX từ URL (ví dụ `./yolo11n_seg_clothes.onnx`) hoặc ArrayBuffer.
   */
  async loadModel(source) {
    const ort = window.ort;
    if (!ort) {
      throw new Error("Không tải được thư viện onnxruntime-web.");
    }

    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1; // Tương thích tốt nhất với iOS Safari không cần COOP/COEP headers

    let modelBytes = source;
    if (typeof source === "string") {
      const resp = await fetch(source, { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`Không tìm thấy file model (${resp.status})`);
      }
      modelBytes = await resp.arrayBuffer();
    }

    const providersToTry = [
      { name: "WebGPU", opts: { executionProviders: ["webgpu", "wasm"] } },
      { name: "WASM SIMD", opts: { executionProviders: ["wasm"] } },
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
        this.providerName = `ONNX (${candidate.name})`;
        return this.providerName;
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error("Không thể khởi tạo ONNX InferenceSession.");
  }

  /**
   * Chuẩn bị ảnh trang phục mẫu (hỗ trợ tự động tách nền trắng/sáng cho ảnh sản phẩm TikTok Shop).
   */
  setGarmentFromSource(imgOrCanvas, autoRemoveWhiteBg = false) {
    const w = imgOrCanvas.naturalWidth || imgOrCanvas.width;
    const h = imgOrCanvas.naturalHeight || imgOrCanvas.height;
    if (!w || !h) return;

    this.garmentSrcCanvas.width = w;
    this.garmentSrcCanvas.height = h;
    this.garmentSrcCtx.clearRect(0, 0, w, h);
    this.garmentSrcCtx.drawImage(imgOrCanvas, 0, 0, w, h);

    const imgData = this.garmentSrcCtx.getImageData(0, 0, w, h);
    if (autoRemoveWhiteBg) {
      removeStudioBackgroundInPlace(imgData.data, w, h);
      this.garmentSrcCtx.putImageData(imgData, 0, 0);
    }
    this.garmentImageData = imgData;
  }

  /**
   * Tạo áo/quần mẫu RGBA dạng vector sắc nét theo lớp & màu.
   */
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
      // Shirt / Jacket
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
      // Pants
      ctx.moveTo(56, 16);
      ctx.lineTo(200, 16);
      ctx.lineTo(212, 244);
      ctx.lineTo(148, 244);
      ctx.lineTo(128, 108);
      ctx.lineTo(108, 244);
      ctx.lineTo(44, 244);
      ctx.closePath();
    } else {
      // Dress
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
   * Tiền xử lý RGBA -> NCHW Float32 [1, 3, 320, 320] trong [0.0..1.0].
   */
  preprocessRGBA(frameData, width, height) {
    const planeSize = INPUT_SIZE * INPUT_SIZE;
    const out = this.inputFloatBuffer;
    const scaleX = width / INPUT_SIZE;
    const scaleY = height / INPUT_SIZE;

    for (let y = 0; y < INPUT_SIZE; y++) {
      const sy = Math.min(height - 1, (y * scaleY) | 0);
      const rowOffset = sy * width;
      const outRow = y * INPUT_SIZE;

      for (let x = 0; x < INPUT_SIZE; x++) {
        const sx = Math.min(width - 1, (x * scaleX) | 0);
        const srcIdx = (rowOffset + sx) * 4;
        const dstIdx = outRow + x;

        out[dstIdx] = frameData[srcIdx] / 255.0;
        out[planeSize + dstIdx] = frameData[srcIdx + 1] / 255.0;
        out[2 * planeSize + dstIdx] = frameData[srcIdx + 2] / 255.0;
      }
    }
    return out;
  }

  /**
   * Xử lý 1 frame ImageData RGBA trực tiếp tại chỗ (in-place).
   * Trả về { detected: boolean, confidence: number, mode: string }.
   */
  async processFrame(frameImageData, targetClassId) {
    if (!this.garmentImageData) {
      return { detected: false, confidence: 0, mode: this.providerName };
    }

    const { data: frameData, width, height } = frameImageData;

    if (!this.session) {
      this._applySmartPreviewOverlay(frameData, width, height, targetClassId);
      return { detected: true, confidence: 1.0, mode: "Smart Preview" };
    }

    const ort = window.ort;
    const inputArr = this.preprocessRGBA(frameData, width, height);
    const inputTensor = new ort.Tensor("float32", inputArr, [
      1,
      3,
      INPUT_SIZE,
      INPUT_SIZE,
    ]);

    const results = await this.session.run({ [this.inputName]: inputTensor });
    const outputKeys = Object.keys(results);
    if (outputKeys.length < 2) {
      return { detected: false, confidence: 0, mode: this.providerName };
    }

    const detsTensor = results[outputKeys[0]];
    const protosTensor = results[outputKeys[1]];

    const best = findBestDetection(detsTensor, targetClassId);
    if (!best) {
      return { detected: false, confidence: 0, mode: this.providerName };
    }

    const { bbox, coeffs, confidence } = best;
    const bx = Math.max(0, Math.floor(clamp01(bbox[0]) * width));
    const by = Math.max(0, Math.floor(clamp01(bbox[1]) * height));
    const x2 = Math.min(width, Math.ceil(clamp01(bbox[2]) * width));
    const y2 = Math.min(height, Math.ceil(clamp01(bbox[3]) * height));
    const bw = Math.max(1, x2 - bx);
    const bh = Math.max(1, y2 - by);

    const warpedGarment = this._resizeGarmentToBox(bw, bh);

    if (this.wasmDecodeAndBlend) {
      const frameBytes = new Uint8Array(
        frameData.buffer,
        frameData.byteOffset,
        frameData.byteLength
      );
      const garmentBytes = new Uint8Array(
        warpedGarment.data.buffer,
        warpedGarment.data.byteOffset,
        warpedGarment.data.byteLength
      );
      this.wasmDecodeAndBlend(
        frameBytes,
        width,
        height,
        protosTensor.data,
        protosTensor.dims[2],
        protosTensor.dims[3],
        coeffs,
        garmentBytes,
        bx,
        by,
        bw,
        bh
      );
    } else {
      decodeMaskAndBlend(
        frameData,
        width,
        height,
        protosTensor,
        coeffs,
        warpedGarment.data,
        bx,
        by,
        bw,
        bh
      );
    }

    return { detected: true, confidence, mode: this.providerName };
  }

  _resizeGarmentToBox(bw, bh) {
    this.garmentResizeCanvas.width = bw;
    this.garmentResizeCanvas.height = bh;
    this.garmentResizeCtx.clearRect(0, 0, bw, bh);
    this.garmentResizeCtx.drawImage(this.garmentSrcCanvas, 0, 0, bw, bh);
    return this.garmentResizeCtx.getImageData(0, 0, bw, bh);
  }

  /**
   * Chế độ thử nghiệm trực quan trên iPhone trước khi nạp file `.onnx` fine-tuned.
   */
  _applySmartPreviewOverlay(frameData, width, height, targetClassId) {
    let rx, ry, rw, rh;
    if (targetClassId === 1) {
      // Quần
      rx = 0.28;
      ry = 0.48;
      rw = 0.44;
      rh = 0.46;
    } else if (targetClassId === 3) {
      // Váy
      rx = 0.24;
      ry = 0.26;
      rw = 0.52;
      rh = 0.62;
    } else {
      // Áo thun / Áo khoác
      rx = 0.22;
      ry = 0.24;
      rw = 0.56;
      rh = 0.44;
    }

    const bx = Math.floor(rx * width);
    const by = Math.floor(ry * height);
    const bw = Math.max(1, Math.floor(rw * width));
    const bh = Math.max(1, Math.floor(rh * height));

    const warped = this._resizeGarmentToBox(bw, bh).data;
    for (let y = 0; y < bh && by + y < height; y++) {
      for (let x = 0; x < bw && bx + x < width; x++) {
        const gIdx = (y * bw + x) * 4;
        const ga = warped[gIdx + 3];
        if (ga <= 20) continue;

        const alpha = (ga / 255.0) * 0.88;
        const fIdx = ((by + y) * width + (bx + x)) * 4;
        frameData[fIdx] = warped[gIdx] * alpha + frameData[fIdx] * (1.0 - alpha);
        frameData[fIdx + 1] =
          warped[gIdx + 1] * alpha + frameData[fIdx + 1] * (1.0 - alpha);
        frameData[fIdx + 2] =
          warped[gIdx + 2] * alpha + frameData[fIdx + 2] * (1.0 - alpha);
      }
    }
  }
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
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

  const cx = data[bestIdx] / INPUT_SIZE;
  const cy = data[anchors + bestIdx] / INPUT_SIZE;
  const w = data[2 * anchors + bestIdx] / INPUT_SIZE;
  const h = data[3 * anchors + bestIdx] / INPUT_SIZE;
  const bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];

  const coeffs = new Float32Array(NUM_MASK_COEFFS);
  const coeffBase = (4 + numClasses) * anchors;
  for (let m = 0; m < NUM_MASK_COEFFS; m++) {
    coeffs[m] = data[coeffBase + m * anchors + bestIdx];
  }

  return { bbox, coeffs, confidence: bestConf };
}

function decodeMaskAndBlend(
  frameData,
  outW,
  outH,
  protosTensor,
  coeffs,
  garmentData,
  bx,
  by,
  bw,
  bh
) {
  const pDims = protosTensor.dims;
  if (!pDims || pDims.length < 4 || pDims[1] < NUM_MASK_COEFFS) return;

  const mh = pDims[2];
  const mw = pDims[3];
  if (mh === 0 || mw === 0) return;

  const protos = protosTensor.data;
  const planeSize = mh * mw;
  const maxY = Math.min(outH, by + bh);
  const maxX = Math.min(outW, bx + bw);

  for (let y = by; y < maxY; y++) {
    const py = Math.min(mh - 1, ((y / outH) * mh) | 0);
    const pRow = py * mw;
    const gRow = (y - by) * bw;
    const fRow = y * outW;

    for (let x = bx; x < maxX; x++) {
      const gIdx = (gRow + (x - bx)) * 4;
      const ga = garmentData[gIdx + 3];
      if (ga <= 20) continue;

      const px = Math.min(mw - 1, ((x / outW) * mw) | 0);
      const pOffset = pRow + px;

      let sum = 0.0;
      for (let c = 0; c < NUM_MASK_COEFFS; c++) {
        sum += coeffs[c] * protos[c * planeSize + pOffset];
      }
      const mVal = 1.0 / (1.0 + Math.exp(-sum));
      if (mVal <= 0.5) continue;

      const alpha = (ga / 255.0) * mVal;
      const fIdx = (fRow + x) * 4;
      frameData[fIdx] = garmentData[gIdx] * alpha + frameData[fIdx] * (1.0 - alpha);
      frameData[fIdx + 1] =
        garmentData[gIdx + 1] * alpha + frameData[fIdx + 1] * (1.0 - alpha);
      frameData[fIdx + 2] =
        garmentData[gIdx + 2] * alpha + frameData[fIdx + 2] * (1.0 - alpha);
    }
  }
}

/**
 * Tách nền trắng/sáng đơn giản cho ảnh sản phẩm TikTok Shop (JPEG) để ghép đẹp hơn lên camera.
 */
function removeStudioBackgroundInPlace(data, width, height) {
  // Lấy màu nền trung bình từ 4 góc ảnh
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

  // Chỉ áp dụng tách nền nếu viền góc sáng
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
