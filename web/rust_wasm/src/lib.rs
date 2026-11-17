use wasm_bindgen::prelude::*;

const NUM_MASK_COEFFS: usize = 32;

#[inline]
fn blend(bg: u8, fg: u8, alpha: f32) -> u8 {
    ((fg as f32 * alpha) + (bg as f32 * (1.0 - alpha))).clamp(0.0, 255.0) as u8
}

/// Giải mã proto masks YOLO11n-seg (32 channels) và blend áo mẫu vào vùng mask trên frame RGBA.
/// Build bằng `wasm-pack build --target web --release` để tăng tốc trên trình duyệt.
#[wasm_bindgen]
pub fn decode_and_blend_wasm(
    frame_rgba: &mut [u8],
    out_w: u32,
    out_h: u32,
    protos: &[f32],
    proto_h: usize,
    proto_w: usize,
    coeffs: &[f32],
    garment_rgba: &[u8],
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
) {
    if out_w == 0
        || out_h == 0
        || proto_h == 0
        || proto_w == 0
        || bw == 0
        || bh == 0
        || coeffs.len() < NUM_MASK_COEFFS
    {
        return;
    }

    let plane_size = proto_h * proto_w;
    if protos.len() < NUM_MASK_COEFFS * plane_size {
        return;
    }

    let expected_frame = (out_w as usize) * (out_h as usize) * 4;
    let expected_garment = (bw as usize) * (bh as usize) * 4;
    if frame_rgba.len() < expected_frame || garment_rgba.len() < expected_garment {
        return;
    }

    let max_y = (by + bh).min(out_h);
    let max_x = (bx + bw).min(out_w);

    for y in by..max_y {
        let py = (((y as f32 / out_h as f32) * proto_h as f32) as usize).min(proto_h - 1);
        let p_row = py * proto_w;
        let g_row = (y - by) as usize * bw as usize;
        let f_row = y as usize * out_w as usize;

        for x in bx..max_x {
            let g_idx = (g_row + (x - bx) as usize) * 4;
            let ga = garment_rgba[g_idx + 3];
            if ga <= 20 {
                continue;
            }

            let px = (((x as f32 / out_w as f32) * proto_w as f32) as usize).min(proto_w - 1);
            let p_offset = p_row + px;

            let mut sum = 0.0f32;
            for c in 0..NUM_MASK_COEFFS {
                sum += coeffs[c] * protos[c * plane_size + p_offset];
            }
            let m_val = 1.0 / (1.0 + (-sum).exp());
            if m_val <= 0.5 {
                continue;
            }

            let alpha = (ga as f32 / 255.0) * m_val;
            let f_idx = (f_row + x as usize) * 4;
            frame_rgba[f_idx] = blend(frame_rgba[f_idx], garment_rgba[g_idx], alpha);
            frame_rgba[f_idx + 1] = blend(frame_rgba[f_idx + 1], garment_rgba[g_idx + 1], alpha);
            frame_rgba[f_idx + 2] = blend(frame_rgba[f_idx + 2], garment_rgba[g_idx + 2], alpha);
        }
    }
}
