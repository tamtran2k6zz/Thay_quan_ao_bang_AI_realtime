use image::{imageops::FilterType, RgbaImage};
use jni::objects::{JByteArray, JByteBuffer, JClass, JString};
use jni::sys::{jbyte, jint, jlong};
use jni::JNIEnv;
use ndarray::{Array4, IxDyn};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::slice;

const INPUT_SIZE: usize = 320;
const CONF_THRESHOLD: f32 = 0.45;
const NUM_MASK_COEFFS: usize = 32;

/// Engine xử lý thay quần áo real-time từ frame camera điện thoại.
pub struct TryOnEngine {
    session: Session,
}

impl TryOnEngine {
    /// Khởi tạo ONNX Runtime Session từ file model đã train trên Google Colab.
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Nhận frame RGBA từ camera, tách vùng quần áo (ví dụ class 0: shirt) và ghép áo mới.
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

        let expected_frame_len = checked_rgba_len(width, height)
            .ok_or("Frame buffer dimensions overflow")?;
        let expected_garment_len = checked_rgba_len(garment_w, garment_h)
            .ok_or("Garment buffer dimensions overflow")?;

        if frame_rgba.len() < expected_frame_len {
            return Err("Camera frame buffer smaller than width * height * 4".into());
        }
        if garment_rgba.len() < expected_garment_len {
            return Err("Garment buffer smaller than garment_w * garment_h * 4".into());
        }

        let frame_slice = &mut frame_rgba[..expected_frame_len];
        let garment_slice = &garment_rgba[..expected_garment_len];

        // 1. Resize frame camera về 320x320 NCHW float32 [0.0..1.0]
        let cam_img = RgbaImage::from_raw(width, height, frame_slice.to_vec())
            .ok_or("Invalid camera buffer size")?;
        let resized = image::imageops::resize(
            &cam_img,
            INPUT_SIZE as u32,
            INPUT_SIZE as u32,
            FilterType::Nearest,
        );

        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE, INPUT_SIZE));
        for (x, y, pixel) in resized.enumerate_pixels() {
            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }

        // 2. Chạy suy luận AI (YOLO11n-seg)
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self.session.run(ort::inputs![input_tensor]?)?;

        // output0: [1, 4 + num_classes + 32, num_anchors]
        // output1 (proto masks): [1, 32, mask_h, mask_w]
        let dets = outputs[0].try_extract_array::<f32>()?;
        let protos = outputs[1].try_extract_array::<f32>()?;

        // 3. Tìm đối tượng quần áo có confidence cao nhất khớp với target_class_id
        if let Some((bbox, mask_coeffs)) = find_best_detection(&dets, target_class_id) {
            let mask = decode_mask(&protos, &mask_coeffs, bbox, width, height);

            // 4. Resize áo mẫu theo bounding box và blend vào vùng mask trên frame gốc
            let bx = (bbox[0].clamp(0.0, 1.0) * width as f32) as u32;
            let by = (bbox[1].clamp(0.0, 1.0) * height as f32) as u32;
            let x2 = (bbox[2].clamp(0.0, 1.0) * width as f32) as u32;
            let y2 = (bbox[3].clamp(0.0, 1.0) * height as f32) as u32;
            let bw = x2.saturating_sub(bx).max(1);
            let bh = y2.saturating_sub(by).max(1);

            if let Some(garment_img) =
                RgbaImage::from_raw(garment_w, garment_h, garment_slice.to_vec())
            {
                let warped_garment =
                    image::imageops::resize(&garment_img, bw, bh, FilterType::Triangle);

                for y in by..(by + bh).min(height) {
                    for x in bx..(bx + bw).min(width) {
                        let m_val = mask[(y * width + x) as usize];
                        if m_val > 0.5 {
                            let gp = warped_garment.get_pixel(x - bx, y - by);
                            if gp[3] > 20 {
                                let idx = ((y * width + x) * 4) as usize;
                                let alpha = (gp[3] as f32 / 255.0) * m_val;
                                frame_slice[idx] = blend(frame_slice[idx], gp[0], alpha);
                                frame_slice[idx + 1] = blend(frame_slice[idx + 1], gp[1], alpha);
                                frame_slice[idx + 2] = blend(frame_slice[idx + 2], gp[2], alpha);
                            }
                        }
                    }
                }
            }
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
    let mut best_result = None;

    for i in 0..anchors {
        let conf = dets[IxDyn(&[0, 4 + target_class, i])];
        if conf > best_conf {
            best_conf = conf;
            let cx = dets[IxDyn(&[0, 0, i])] / INPUT_SIZE as f32;
            let cy = dets[IxDyn(&[0, 1, i])] / INPUT_SIZE as f32;
            let w = dets[IxDyn(&[0, 2, i])] / INPUT_SIZE as f32;
            let h = dets[IxDyn(&[0, 3, i])] / INPUT_SIZE as f32;
            let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];

            let mut coeffs = [0.0f32; NUM_MASK_COEFFS];
            for m in 0..NUM_MASK_COEFFS {
                coeffs[m] = dets[IxDyn(&[0, 4 + num_classes + m, i])];
            }
            best_result = Some((bbox, coeffs));
        }
    }
    best_result
}

fn decode_mask(
    protos: &ndarray::ArrayViewD<f32>,
    coeffs: &[f32; NUM_MASK_COEFFS],
    bbox: [f32; 4],
    out_w: u32,
    out_h: u32,
) -> Vec<f32> {
    let shape = protos.shape();
    let mut mask = vec![0.0f32; (out_w as usize) * (out_h as usize)];
    if shape.len() < 4 || shape[1] < NUM_MASK_COEFFS {
        return mask;
    }
    let mh = shape[2];
    let mw = shape[3];
    if mh == 0 || mw == 0 || out_w == 0 || out_h == 0 {
        return mask;
    }

    for y in 0..out_h {
        let ny = y as f32 / out_h as f32;
        if ny < bbox[1] || ny > bbox[3] {
            continue;
        }
        let py = ((ny * mh as f32) as usize).min(mh - 1);

        for x in 0..out_w {
            let nx = x as f32 / out_w as f32;
            if nx < bbox[0] || nx > bbox[2] {
                continue;
            }
            let px = ((nx * mw as f32) as usize).min(mw - 1);

            let mut sum = 0.0f32;
            for c in 0..NUM_MASK_COEFFS {
                sum += coeffs[c] * protos[IxDyn(&[0, c, py, px])];
            }
            // Sigmoid activation
            mask[(y * out_w + x) as usize] = 1.0 / (1.0 + (-sum).exp());
        }
    }
    mask
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
