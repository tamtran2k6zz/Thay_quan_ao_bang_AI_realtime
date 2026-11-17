#ifndef RUST_TRYON_ENGINE_H
#define RUST_TRYON_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle tới Rust TryOnEngine.
 */
typedef struct TryOnEngine TryOnEngine;

/**
 * Khởi tạo ONNX Runtime session từ đường dẫn file model .onnx.
 * Trả về NULL nếu thất bại. Phải giải phóng bằng tryon_free().
 */
TryOnEngine *tryon_init(const char *model_path);

/**
 * Xử lý một frame RGBA8888 từ camera (chỉnh sửa trực tiếp tại chỗ trên frame_ptr).
 *
 * @param engine_ptr      Con trỏ engine từ tryon_init().
 * @param frame_ptr       Buffer RGBA8888 của frame camera (kích thước width * height * 4 bytes).
 * @param width           Chiều rộng frame camera (px).
 * @param height          Chiều cao frame camera (px).
 * @param garment_ptr     Buffer RGBA8888 của ảnh quần áo mẫu (nền trong suốt PNG).
 * @param garment_w       Chiều rộng ảnh quần áo mẫu (px).
 * @param garment_h       Chiều cao ảnh quần áo mẫu (px).
 * @param target_class_id ID lớp cần thay (0: shirt, 1: pants, 2: jacket, 3: dress).
 * @return 0 nếu thành công, -1 nếu tham số không hợp lệ, -2 nếu lỗi suy luận ONNX.
 */
int32_t tryon_process_frame(
    TryOnEngine *engine_ptr,
    uint8_t *frame_ptr,
    uint32_t width,
    uint32_t height,
    const uint8_t *garment_ptr,
    uint32_t garment_w,
    uint32_t garment_h,
    size_t target_class_id
);

/**
 * Giải phóng bộ nhớ TryOnEngine.
 */
void tryon_free(TryOnEngine *engine_ptr);

#ifdef __cplusplus
}
#endif

#endif /* RUST_TRYON_ENGINE_H */
