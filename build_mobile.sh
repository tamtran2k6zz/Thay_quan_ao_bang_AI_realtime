#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== Rust Mobile TryOn Engine Build Script ==="

if ! command -v cargo >/dev/null 2>&1; then
  echo "Lỗi: Chưa cài đặt 'cargo'. Cài Rust tại https://rustup.rs/"
  exit 1
fi

BUILD_ANDROID=true
BUILD_IOS=false

if [[ "$(uname -s)" == "Darwin" ]]; then
  BUILD_IOS=true
fi

# ----------------------------------------------------------------------------
# 1. Build Android (.so) -> android/app/src/main/jniLibs/
# ----------------------------------------------------------------------------
if [[ "$BUILD_ANDROID" == "true" ]]; then
  if ! cargo ndk --version >/dev/null 2>&1; then
    echo "Đang cài đặt cargo-ndk..."
    cargo install cargo-ndk
  fi

  rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android

  JNILIBS_DIR="$PROJECT_ROOT/android/app/src/main/jniLibs"
  mkdir -p "$JNILIBS_DIR"

  echo "Đang build librust_tryon_engine.so cho Android..."
  cargo ndk \
    -t arm64-v8a \
    -t armeabi-v7a \
    -t x86_64 \
    -o "$JNILIBS_DIR" \
    build --release

  echo "Android .so đã xuất tại: $JNILIBS_DIR"
fi

# ----------------------------------------------------------------------------
# 2. Build iOS (.a & optional XCFramework) trên macOS
# ----------------------------------------------------------------------------
if [[ "$BUILD_IOS" == "true" ]]; then
  echo "Phát hiện macOS -> Tiến hành build staticlib (.a) cho iOS..."
  rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios

  cargo build --release --target aarch64-apple-ios
  cargo build --release --target aarch64-apple-ios-sim
  cargo build --release --target x86_64-apple-ios

  IOS_LIBS_DIR="$PROJECT_ROOT/ios/Libs"
  mkdir -p "$IOS_LIBS_DIR/device" "$IOS_LIBS_DIR/sim"

  cp "$PROJECT_ROOT/target/aarch64-apple-ios/release/librust_tryon_engine.a" \
     "$IOS_LIBS_DIR/device/librust_tryon_engine.a"

  lipo -create \
    "$PROJECT_ROOT/target/aarch64-apple-ios-sim/release/librust_tryon_engine.a" \
    "$PROJECT_ROOT/target/x86_64-apple-ios/release/librust_tryon_engine.a" \
    -output "$IOS_LIBS_DIR/sim/librust_tryon_engine.a"

  if command -v xcodebuild >/dev/null 2>&1; then
    XCFRAMEWORK_OUT="$IOS_LIBS_DIR/RustTryOnEngine.xcframework"
    rm -rf "$XCFRAMEWORK_OUT"
    xcodebuild -create-xcframework \
      -library "$IOS_LIBS_DIR/device/librust_tryon_engine.a" -headers "$PROJECT_ROOT/include" \
      -library "$IOS_LIBS_DIR/sim/librust_tryon_engine.a" -headers "$PROJECT_ROOT/include" \
      -output "$XCFRAMEWORK_OUT"
    echo "Đã tạo XCFramework tại: $XCFRAMEWORK_OUT"
  fi
else
  echo "Bỏ qua build iOS (.a) vì hệ điều hành hiện tại không phải macOS."
fi
