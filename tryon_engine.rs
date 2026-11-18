use jni::objects::{JByteArray, JByteBuffer, JClass, JString};
use jni::sys::{jbyte, jint, jlong};
use jni::JNIEnv;
use ndarray::{Array4, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

const INPUT_SIZE: usize = 320;
const CONF_THRESHOLD: f32 = 0.45;
const NUM_MASK_COEFFS: usize = 32;
const BBOX_EMA_ALPHA: f32 = 0.45;

/// Engine xử lý thay quần áo real-time từ frame camera điện thoại (tối ưu zero-alloc & 80x80 mask decode).
pub struct TryOnEngine {
    session: Session,
    input_tensor_buf: Array4<f32>,
    mask_grid: Vec<f32>,
    smoothed_bbox: Option<[f32; 4]>,
    last_target_class: Option<usize>,
}

impl TryOnEngine {
    /// Khởi tạo ONNX Runtime Session từ file model đã train trên Google Colab.
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_tensor_buf: Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE)),
            mask_grid: vec![0.0f32; 80 * 80],
            smoothed_bbox: None,
            last_target_class: None,
        })
    }

    /// Nhận frame RGBA từ camera, tách vùng quần áo (0: shirt, 1: pants, 2: jacket, 3: dress) và ghép áo mới.
    pub fn process_frame(
        &mut self,
        frame_rgba: &mut [u8],
        width: u32,
        height: u32,
        garment_rgba: &[u8],
        garment_w: u32,
        garment_h: u32,
        target_class_id: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if width == 0 || height == 0 || garment_w == 0 || garment_h == 0 {
            return Err("Invalid zero frame or garment dimension".into());
        }

        let expected_frame_len =
            checked_rgba_len(width, height).ok_or("Frame buffer dimensions overflow")?;
        let expected_garment_len =
            checked_rgba_len(garment_w, garment_h).ok_or("Garment buffer dimensions overflow")?;

        if frame_rgba.len() < expected_frame_len {
            return Err("Camera frame buffer smaller than width * height * 4".into());
        }
        if garment_rgba.len() < expected_garment_len {
            return Err("Garment buffer smaller than garment_w * garment_h * 4".into());
        }

        if self.last_target_class != Some(target_class_id) {
            self.smoothed_bbox = None;
            self.last_target_class = Some(target_class_id);
        }

        let frame_slice = &mut frame_rgba[..expected_frame_len];
        let garment_slice = &garment_rgba[..expected_garment_len];

        // 1. Lấy mẫu trực tiếp RGBA -> NCHW [1, 3, 320, 320] float32 [0.0..1.0] (không clone/resize trung gian)
        {
            let in_data = self
                .input_tensor_buf
                .as_slice_mut()
                .ok_or("Non-contiguous input tensor buffer")?;
            let plane = INPUT_SIZE * INPUT_SIZE;
            let w_usize = width as usize;
            let h_usize = height as usize;
            let inv_255 = 1.0f32 / 255.0;

            for y in 0..INPUT_SIZE {
                let sy = (y * h_usize / INPUT_SIZE).min(h_usize - 1);
                let row_base = sy * w_usize;
                let out_row = y * INPUT_SIZE;

                for x in 0..INPUT_SIZE {
                    let sx = (x * w_usize / INPUT_SIZE).min(w_usize - 1);
                    let p_idx = (row_base + sx) * 4;
                    let dst = out_row + x;
                    in_data[dst] = frame_slice[p_idx] as f32 * inv_255;
                    in_data[plane + dst] = frame_slice[p_idx + 1] as f32 * inv_255;
                    in_data[2 * plane + dst] = frame_slice[p_idx + 2] as f32 * inv_255;
                }
            }
        }

        // 2. Chạy suy luận AI (YOLO11n-seg) với zero-copy TensorRef
        let input_tensor = TensorRef::from_array_view(self.input_tensor_buf.view())?;
        let outputs = self.session.run(ort::inputs![input_tensor]?)?;

        // output0: [1, 4 + num_classes + 32, num_anchors]
        // output1 (proto masks): [1, 32, mh, mw] (mặc định 80x80)
        let dets = outputs[0].try_extract_array::<f32>()?;
        let protos = outputs[1].try_extract_array::<f32>()?;

        // 3. Tìm đối tượng quần áo tốt nhất và giải mã mask ở đúng độ phân giải prototype (80x80)
        if let Some((raw_bbox, mask_coeffs)) = find_best_detection(&dets, target_class_id) {
            let bbox = smooth_bbox(self.smoothed_bbox, raw_bbox);
            self.smoothed_bbox = Some(bbox);

            let proto_shape = protos.shape();
            if proto_shape.len() < 4 || proto_shape[1] < NUM_MASK_COEFFS {
                return Ok(());
            }
            let mh = proto_shape[2];
            let mw = proto_shape[3];
            if mh == 0 || mw == 0 {
                return Ok(());
            }

            let grid_len = mh * mw;
            if self.mask_grid.len() != grid_len {
                self.mask_grid.resize(grid_len, 0.0);
            }
            decode_proto_mask_bbox(&protos, &mask_coeffs, bbox, mh, mw, &mut self.mask_grid);

            // 4. Blend áo mẫu trực tiếp vào vùng mask bằng nội suy song tuyến (không cấp phát resize buffer)
            let bx = (bbox[0].clamp(0.0, 1.0) * width as f32).floor() as u32;
            let by = (bbox[1].clamp(0.0, 1.0) * height as f32).floor() as u32;
            let x2 = (bbox[2].clamp(0.0, 1.0) * width as f32).ceil().min(width as f32) as u32;
            let y2 = (bbox[3].clamp(0.0, 1.0) * height as f32).ceil().min(height as f32) as u32;
            let bw = x2.saturating_sub(bx).max(1);
            let bh = y2.saturating_sub(by).max(1);

            let inv_out_w = 1.0 / width as f32;
            let inv_out_h = 1.0 / height as f32;
            let scale_gx = garment_w as f32 / bw as f32;
            let scale_gy = garment_h as f32 / bh as f32;
            let max_gx = (garment_w - 1) as f32;
            let max_gy = (garment_h - 1) as f32;

            for y in by..y2 {
                let my = ((y as f32 + 0.5) * inv_out_h) * mh as f32 - 0.5;
                let gy_f = (((y - by) as f32 + 0.5) * scale_gy - 0.5).clamp(0.0, max_gy);
                let row_off = (y * width) as usize;

                for x in bx..x2 {
                    let mx = ((x as f32 + 0.5) * inv_out_w) * mw as f32 - 0.5;
                    let m_val = sample_bilinear_mask(&self.mask_grid, mw, mh, mx, my);
                    if m_val <= 0.5 {
                        continue;
                    }

                    let gx_f = (((x - bx) as f32 + 0.5) * scale_gx - 0.5).clamp(0.0, max_gx);
                    let gp = sample_bilinear_rgba(
                        garment_slice,
                        garment_w as usize,
                        garment_h as usize,
                        gx_f,
                        gy_f,
                    );
                    if gp[3] > 20 {
                        let idx = (row_off + x as usize) * 4;
                        let alpha = (gp[3] as f32 * (1.0 / 255.0)) * m_val;
                        frame_slice[idx] = blend(frame_slice[idx], gp[0], alpha);
                        frame_slice[idx + 1] = blend(frame_slice[idx + 1], gp[1], alpha);
                        frame_slice[idx + 2] = blend(frame_slice[idx + 2], gp[2], alpha);
                    }
                }
            }
        } else {
            self.smoothed_bbox = None;
        }

        Ok(())
    }
}

#[inline]
fn checked_rgba_len(width: u32, height: u32) -> Option<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
}

#[inline]
fn blend(bg: u8, fg: u8, alpha: f32) -> u8 {
    ((fg as f32 * alpha) + (bg as f32 * (1.0 - alpha))).clamp(0.0, 255.0) as u8
}

#[inline]
fn smooth_bbox(prev: Option<[f32; 4]>, curr: [f32; 4]) -> [f32; 4] {
    match prev {
        Some(p) => {
            let a = BBOX_EMA_ALPHA;
            let b = 1.0 - a;
            [
                a * curr[0] + b * p[0],
                a * curr[1] + b * p[1],
                a * curr[2] + b * p[2],
                a * curr[3] + b * p[3],
            ]
        }
        None => curr,
    }
}

fn find_best_detection(
    dets: &ndarray::ArrayViewD<f32>,
    target_class: usize,
) -> Option<([f32; 4], [f32; NUM_MASK_COEFFS])> {
    let shape = dets.shape(); // [1, 4 + C + 32, N]
    if shape.len() < 3 {
        return None;
    }
    let channels = shape[1];
    let anchors = shape[2];
    if channels <= 4 + NUM_MASK_COEFFS {
        return None;
    }
    let num_classes = channels - 4 - NUM_MASK_COEFFS;
    if target_class >= num_classes {
        return None;
    }

    let mut best_conf = CONF_THRESHOLD;
    let mut best_idx: Option<usize> = None;

    for i in 0..anchors {
        let conf = dets[IxDyn(&[0, 4 + target_class, i])];
        if conf > best_conf {
            best_conf = conf;
            best_idx = Some(i);
        }
    }

    let i = best_idx?;
    let inv_input = 1.0f32 / INPUT_SIZE as f32;
    let cx = dets[IxDyn(&[0, 0, i])] * inv_input;
    let cy = dets[IxDyn(&[0, 1, i])] * inv_input;
    let w = dets[IxDyn(&[0, 2, i])] * inv_input;
    let h = dets[IxDyn(&[0, 3, i])] * inv_input;
    let bbox = [
        (cx - w * 0.5).clamp(0.0, 1.0),
        (cy - h * 0.5).clamp(0.0, 1.0),
        (cx + w * 0.5).clamp(0.0, 1.0),
        (cy + h * 0.5).clamp(0.0, 1.0),
    ];

    let mut coeffs = [0.0f32; NUM_MASK_COEFFS];
    for m in 0..NUM_MASK_COEFFS {
        coeffs[m] = dets[IxDyn(&[0, 4 + num_classes + m, i])];
    }
    Some((bbox, coeffs))
}

fn decode_proto_mask_bbox(
    protos: &ndarray::ArrayViewD<f32>,
    coeffs: &[f32; NUM_MASK_COEFFS],
    bbox: [f32; 4],
    mh: usize,
    mw: usize,
    mask_grid: &mut [f32],
) {
    mask_grid.fill(0.0);

    let px1 = ((bbox[0] * mw as f32).floor() as isize - 1).max(0) as usize;
    let py1 = ((bbox[1] * mh as f32).floor() as isize - 1).max(0) as usize;
    let px2 = ((bbox[2] * mw as f32).ceil() as usize + 1).min(mw);
    let py2 = ((bbox[3] * mh as f32).ceil() as usize + 1).min(mh);

    for py in py1..py2 {
        let row_off = py * mw;
        for px in px1..px2 {
            let mut sum = 0.0f32;
            for c in 0..NUM_MASK_COEFFS {
                sum += coeffs[c] * protos[IxDyn(&[0, c, py, px])];
            }
            mask_grid[row_off + px] = 1.0 / (1.0 + (-sum).exp());
        }
    }
}

#[inline]
fn sample_bilinear_mask(grid: &[f32], mw: usize, mh: usize, fx: f32, fy: f32) -> f32 {
    let x = fx.clamp(0.0, (mw - 1) as f32);
    let y = fy.clamp(0.0, (mh - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(mw - 1);
    let y1 = (y0 + 1).min(mh - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let v00 = grid[y0 * mw + x0];
    let v10 = grid[y0 * mw + x1];
    let v01 = grid[y1 * mw + x0];
    let v11 = grid[y1 * mw + x1];

    let top = v00 + (v10 - v00) * dx;
    let bot = v01 + (v11 - v01) * dx;
    top + (bot - top) * dy
}

#[inline]
fn sample_bilinear_rgba(
    rgba: &[u8],
    w: usize,
    h: usize,
    fx: f32,
    fy: f32,
) -> [u8; 4] {
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let dx = fx - x0 as f32;
    let dy = fy - y0 as f32;

    let i00 = (y0 * w + x0) * 4;
    let i10 = (y0 * w + x1) * 4;
    let i01 = (y1 * w + x0) * 4;
    let i11 = (y1 * w + x1) * 4;

    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;

    [
        (rgba[i00] as f32 * w00 + rgba[i10] as f32 * w10 + rgba[i01] as f32 * w01 + rgba[i11] as f32 * w11) as u8,
        (rgba[i00 + 1] as f32 * w00 + rgba[i10 + 1] as f32 * w10 + rgba[i01 + 1] as f32 * w01 + rgba[i11 + 1] as f32 * w11) as u8,
        (rgba[i00 + 2] as f32 * w00 + rgba[i10 + 2] as f32 * w10 + rgba[i01 + 2] as f32 * w01 + rgba[i11 + 2] as f32 * w11) as u8,
        (rgba[i00 + 3] as f32 * w00 + rgba[i10 + 3] as f32 * w10 + rgba[i01 + 3] as f32 * w01 + rgba[i11 + 3] as f32 * w11) as u8,
    ]
}

// ============================================================================
// C-ABI FFI cho iOS (Swift) và C/C++ gọi trực tiếp từ Camera
// ============================================================================

#[no_mangle]
pub extern "C" fn tryon_init(model_path: *const c_char) -> *mut TryOnEngine {
    if model_path.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(model_path) };
    let Ok(path_str) = c_str.to_str() else {
        return std::ptr::null_mut();
    };

    match TryOnEngine::new(path_str) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tryon_process_frame(
    engine_ptr: *mut TryOnEngine,
    frame_ptr: *mut u8,
    width: u32,
    height: u32,
    garment_ptr: *const u8,
    garment_w: u32,
    garment_h: u32,
    target_class_id: usize,
) -> i32 {
    if engine_ptr.is_null()
        || frame_ptr.is_null()
        || garment_ptr.is_null()
        || width == 0
        || height == 0
        || garment_w == 0
        || garment_h == 0
    {
        return -1;
    }

    let Some(frame_len) = checked_rgba_len(width, height) else {
        return -1;
    };
    let Some(garment_len) = checked_rgba_len(garment_w, garment_h) else {
        return -1;
    };

    let engine = unsafe { &mut *engine_ptr };
    let frame = unsafe { slice::from_raw_parts_mut(frame_ptr, frame_len) };
    let garment = unsafe { slice::from_raw_parts(garment_ptr, garment_len) };

    match engine.process_frame(frame, width, height, garment, garment_w, garment_h, target_class_id)
    {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn tryon_free(engine_ptr: *mut TryOnEngine) {
    if !engine_ptr.is_null() {
        unsafe { drop(Box::from_raw(engine_ptr)) };
    }
}

// ============================================================================
// Android JNI cho Kotlin (CameraX RGBA_8888 DirectByteBuffer & ByteArray)
// Package: com.tryon.app, Class: TryOnNative
// ============================================================================

#[no_mangle]
pub extern "system" fn Java_com_tryon_app_TryOnNative_init<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    model_path: JString<'local>,
) -> jlong {
    let path: String = match env.get_string(&model_path) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };

    match TryOnEngine::new(&path) {
        Ok(engine) => Box::into_raw(Box::new(engine)) as jlong,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_tryon_app_TryOnNative_processFrameDirect<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    engine_ptr: jlong,
    frame_buffer: JByteBuffer<'local>,
    width: jint,
    height: jint,
    garment_buffer: JByteBuffer<'local>,
    garment_w: jint,
    garment_h: jint,
    target_class_id: jint,
) -> jint {
    if engine_ptr == 0
        || width <= 0
        || height <= 0
        || garment_w <= 0
        || garment_h <= 0
        || target_class_id < 0
    {
        return -1;
    }

    let Some(frame_len) = checked_rgba_len(width as u32, height as u32) else {
        return -1;
    };
    let Some(garment_len) = checked_rgba_len(garment_w as u32, garment_h as u32) else {
        return -1;
    };

    let Ok(frame_ptr) = env.get_direct_buffer_address(&frame_buffer) else {
        return -1;
    };
    let Ok(frame_cap) = env.get_direct_buffer_capacity(&frame_buffer) else {
        return -1;
    };
    let Ok(garment_ptr) = env.get_direct_buffer_address(&garment_buffer) else {
        return -1;
    };
    let Ok(garment_cap) = env.get_direct_buffer_capacity(&garment_buffer) else {
        return -1;
    };

    if frame_ptr.is_null()
        || garment_ptr.is_null()
        || frame_cap < frame_len
        || garment_cap < garment_len
    {
        return -1;
    }

    let engine = unsafe { &mut *(engine_ptr as *mut TryOnEngine) };
    let frame = unsafe { slice::from_raw_parts_mut(frame_ptr, frame_len) };
    let garment = unsafe { slice::from_raw_parts(garment_ptr, garment_len) };

    match engine.process_frame(
        frame,
        width as u32,
        height as u32,
        garment,
        garment_w as u32,
        garment_h as u32,
        target_class_id as usize,
    ) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_tryon_app_TryOnNative_processFrameByteArray<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
    engine_ptr: jlong,
    frame_array: JByteArray<'local>,
    width: jint,
    height: jint,
    garment_array: JByteArray<'local>,
    garment_w: jint,
    garment_h: jint,
    target_class_id: jint,
) -> jint {
    if engine_ptr == 0
        || width <= 0
        || height <= 0
        || garment_w <= 0
        || garment_h <= 0
        || target_class_id < 0
    {
        return -1;
    }

    let Some(frame_len) = checked_rgba_len(width as u32, height as u32) else {
        return -1;
    };
    let Some(garment_len) = checked_rgba_len(garment_w as u32, garment_h as u32) else {
        return -1;
    };

    let Ok(mut frame_vec) = env.convert_byte_array(&frame_array) else {
        return -1;
    };
    let Ok(garment_vec) = env.convert_byte_array(&garment_array) else {
        return -1;
    };

    if frame_vec.len() < frame_len || garment_vec.len() < garment_len {
        return -1;
    }

    let engine = unsafe { &mut *(engine_ptr as *mut TryOnEngine) };
    if engine
        .process_frame(
            &mut frame_vec[..frame_len],
            width as u32,
            height as u32,
            &garment_vec[..garment_len],
            garment_w as u32,
            garment_h as u32,
            target_class_id as usize,
        )
        .is_err()
    {
        return -2;
    }

    let signed_slice =
        unsafe { slice::from_raw_parts(frame_vec.as_ptr() as *const jbyte, frame_len) };
    if env.set_byte_array_region(&frame_array, 0, signed_slice).is_err() {
        return -3;
    }

    0
}

#[no_mangle]
pub extern "system" fn Java_com_tryon_app_TryOnNative_free<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
    engine_ptr: jlong,
) {
    if engine_ptr != 0 {
        unsafe {
            drop(Box::from_raw(engine_ptr as *mut TryOnEngine));
        }
    }
}
