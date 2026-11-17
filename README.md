# Real-Time Virtual Try-On App (Colab YOLO11n-seg + Rust Core + Android/iOS Camera)

Dự án ứng dụng thay quần áo theo thời gian thực qua camera điện thoại:
- **AI Training / Fine-tuning**: Google Colab (`train_clothes_yolo_colab.ipynb`) dùng `YOLO11n-seg` (`imgsz=320`) xuất sang ONNX.
- **Core Engine (Rust)**: `src/lib.rs` + `Cargo.toml` dùng `ort` (ONNX Runtime), `ndarray`, `image` và `jni`, biên dịch ra thư viện native `.so` (Android) và `.a` (iOS) với cả C-ABI FFI và Android JNI (`DirectByteBuffer`).
- **Android UI (Kotlin + CameraX + Jetpack Compose)**: thư mục `android/`.
- **iOS UI (Swift + AVFoundation + SwiftUI)**: thư mục `ios/` và C header `include/rust_tryon_engine.h`.

---

## Cấu trúc thư mục

```text
Realtime_App_Custom/
├── Cargo.toml
├── src/
│   └── lib.rs                                      # Rust TryOnEngine + C-FFI + Android JNI
├── tryon_engine.rs                                 # Bản sao đồng bộ với src/lib.rs
├── include/
│   └── rust_tryon_engine.h                         # C header cho Swift / Xcode Bridging Header
├── train_clothes_yolo_colab.ipynb                  # Notebook train & export ONNX trên Colab
├── build_mobile.ps1                                # Script build Android .so trên Windows
├── build_mobile.sh                                 # Script build Android .so & iOS .a trên macOS/Linux
├── android/
│   └── app/
│       ├── build.gradle.kts
│       └── src/main/
│           ├── AndroidManifest.xml
│           └── java/com/tryon/app/
│               ├── TryOnNative.kt                  # JNI wrapper + TryOnController
│               └── MainActivity.kt                 # CameraX RGBA_8888 + Jetpack Compose UI
├── ios/
│   ├── TryOnApp-Bridging-Header.h                  # Import rust_tryon_engine.h
│   ├── TryOnEngineWrapper.swift                    # Swift wrapper + Accelerate BGRA->RGBA
│   └── CameraTryOnView.swift                       # AVFoundation camera + SwiftUI
└── web/                                            # Chạy trên iPhone Safari không cần Mac
    ├── index.html                                  # Giao diện Camera iOS PWA + chọn đồ TikTok Shop
    ├── tryon_web_engine.js                         # ONNX Runtime Web (WASM/WebGPU) + Smart Preview
    ├── manifest.webmanifest                        # Add to Home Screen trên Safari iPhone
    ├── serve_https.py                              # HTTPS server tự tạo chứng chỉ SSL LAN
    └── rust_wasm/                                  # (Tùy chọn) Nhân decode/blend Rust WASM
```

---

## Bước 1: Train & Fine-tune AI trên Google Colab

1. Mở file [train_clothes_yolo_colab.ipynb](train_clothes_yolo_colab.ipynb) trên **Google Colab**.
2. Bật GPU: **Runtime → Change runtime type → T4 GPU**.
3. Chuẩn bị dataset phân đoạn quần áo theo chuẩn YOLO Segmentation (`data.yaml` mặc định gồm 4 lớp):
   - `0`: `shirt` (Áo thun)
   - `1`: `pants` (Quần)
   - `2`: `jacket` (Áo khoác)
   - `3`: `dress` (Váy)
4. Chạy toàn bộ notebook để fine-tune `yolo11n-seg.pt` (`imgsz=320`) và xuất ra file `.onnx` (`best.onnx`).
5. Đổi tên file tải về thành **`yolo11n_seg_clothes.onnx`**.

---

## Bước 2: Biên dịch Rust Core sang `.so` (Android) và `.a` (iOS)

### Yêu cầu cài đặt
- [Rustup & Cargo](https://rustup.rs/)
- **Cho Android**: Android NDK (qua Android Studio SDK Manager) và `cargo-ndk`:
  ```bash
  cargo install cargo-ndk
  ```
- **Cho iOS (chỉ trên macOS)**: Xcode + Command Line Tools.

### Chạy script build
- **Trên Windows (PowerShell)** (xuất `.so` vào `android/app/src/main/jniLibs/`):
  ```powershell
  .\build_mobile.ps1
  ```
- **Trên macOS / Linux** (xuất cả Android `.so` và iOS `.a` / `.xcframework`):
  ```bash
  chmod +x build_mobile.sh
  ./build_mobile.sh
  ```

---

## Bước 3: Tích hợp & Chạy ứng dụng Mobile

### Android (Kotlin + CameraX + Compose)
1. Copy model đã train vào thư mục assets:
   ```text
   android/app/src/main/assets/yolo11n_seg_clothes.onnx
   ```
   *(Tùy chọn: có thể thêm ảnh PNG tách nền vào `android/app/src/main/assets/garments/shirt.png`, `pants.png`, `jacket.png`, `dress.png` hoặc chọn ảnh trực tiếp từ máy bằng nút **Chọn PNG** trong app).*
2. Đảm bảo đã có `librust_tryon_engine.so` trong `android/app/src/main/jniLibs/<abi>/` (tạo bởi `build_mobile.ps1` hoặc `build_mobile.sh`).
3. Mở module `android/` trong Android Studio (tạo root `settings.gradle.kts` / `build.gradle.kts` hoặc chép thư mục `app/` vào project Android Studio mới có Kotlin + Compose plugin), Sync Gradle và chạy trên thiết bị thật (hoặc Emulator x86_64).

### iOS (Swift + AVFoundation + SwiftUI)
1. Tạo Xcode project mới (App → SwiftUI, tên `TryOnApp`).
2. Thêm các file trong `ios/` (`TryOnEngineWrapper.swift`, `CameraTryOnView.swift`, `TryOnApp-Bridging-Header.h`) và `include/rust_tryon_engine.h` vào target.
3. Trong **Build Settings** của Xcode:
   - Đặt **Objective-C Bridging Header** trỏ tới `TryOnApp-Bridging-Header.h`.
   - Thêm thư viện `ios/Libs/device/librust_tryon_engine.a` (hoặc `ios/Libs/RustTryOnEngine.xcframework`) vào **Frameworks, Libraries, and Embedded Content**.
   - Nếu `ort` dùng thư viện động/tĩnh ONNX Runtime trên iOS, link thêm `libc++.tbd` và framework hệ thống `Accelerate.framework`, `CoreML.framework`.
4. Thêm `yolo11n_seg_clothes.onnx` vào **Copy Bundle Resources**.
5. Thêm khóa `NSCameraUsageDescription` và `NSPhotoLibraryUsageDescription` trong `Info.plist`.
6. Đặt `CameraTryOnView()` làm root view trong `@main struct TryOnApp: App`.

### iPhone qua Safari (Không cần máy Mac — Web App / WASM / WebGPU)
1. (Tùy chọn sau khi train Colab) Copy file model vào `web/yolo11n_seg_clothes.onnx` (hoặc chọn trực tiếp bằng nút **Nạp .onnx** ngay trên Safari iPhone). Nếu chưa có `.onnx`, web vẫn chạy được ở chế độ **Smart Preview** với 40 ảnh sản phẩm từ `clothes_dataset/`.
2. Chạy HTTPS server trên Windows (iOS Safari bắt buộc HTTPS để mở Camera qua mạng LAN):
   ```powershell
   python web/serve_https.py
   ```
3. Kết nối iPhone và máy Windows chung Wi-Fi, mở **Safari** trên iPhone truy cập địa chỉ `https://<IP-MÁY-WINDOWS>:8443` được in trên terminal → xác nhận tin cậy chứng chỉ tự ký (**Hiển thị chi tiết → Truy cập trang web này**) → bấm **Bật Camera & Bắt đầu**.
4. *(Tùy chọn)* Trên Safari bấm nút **Chia sẻ → Thêm vào Màn hình chính** để chạy toàn màn hình như app iOS.

---

## Ghi chú tối ưu Real-Time
- Input model cố định `320x320` (`INPUT_SIZE = 320` trong `src/lib.rs`) giúp giảm độ trễ suy luận trên CPU/NPU di động.
- Android sử dụng `ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888` với chiến lược `STRATEGY_KEEP_ONLY_LATEST` và `DirectByteBuffer` để tránh copy thừa qua JNI.
- iOS sử dụng `AVCaptureVideoDataOutput` (`kCVPixelFormatType_32BGRA`) và `Accelerate/vImage` SIMD chuyển đổi BGRA → RGBA trước khi gọi `tryon_process_frame`.
