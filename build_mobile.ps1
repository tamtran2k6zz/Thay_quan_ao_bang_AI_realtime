<#
.SYNOPSIS
    Biên dịch Rust TryOn Engine thành thư viện native cho Android (.so).
    (Lưu ý: Target iOS .a yêu cầu chạy trên macOS có cài Xcode SDK bằng build_mobile.sh)
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "=== Rust Mobile TryOn Engine Build Script (Windows) ===" -ForegroundColor Cyan

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Error "Không tìm thấy 'cargo'. Hãy cài Rust tại https://rustup.rs/ trước."
}

if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    Write-Error "Không tìm thấy 'rustup'. Hãy cài Rustup trước."
}

# Kiểm tra cargo-ndk
$hasCargoNdk = cargo ndk --version 2>$null
if (-not $hasCargoNdk) {
    Write-Host "Đang cài đặt cargo-ndk..." -ForegroundColor Yellow
    cargo install cargo-ndk
}

# Cài đặt các Android targets
$AndroidTargets = @(
    "aarch64-linux-android",
    "armv7-linux-androideabi",
    "x86_64-linux-android"
)

Write-Host "Đang kiểm tra Rust Android targets..." -ForegroundColor Yellow
foreach ($target in $AndroidTargets) {
    rustup target add $target
}

if (-not $env:ANDROID_NDK_HOME) {
    Write-Warning "Biến môi trường ANDROID_NDK_HOME chưa được thiết lập. cargo-ndk sẽ tự tìm NDK trong Android SDK mặc định."
}

$JniLibsDir = Join-Path $ProjectRoot "android\app\src\main\jniLibs"
New-Item -ItemType Directory -Force -Path $JniLibsDir | Out-Null

Write-Host "Đang biên dịch librust_tryon_engine.so cho Android (arm64-v8a, armeabi-v7a, x86_64)..." -ForegroundColor Green
cargo ndk `
    -t arm64-v8a `
    -t armeabi-v7a `
    -t x86_64 `
    -o $JniLibsDir `
    build --release

Write-Host ""
Write-Host "Build Android hoàn tất! Các file .so đã được xuất tại:" -ForegroundColor Green
Get-ChildItem -Path $JniLibsDir -Recurse -Filter "*.so" | ForEach-Object {
    Write-Host "  - $($_.FullName)"
}

Write-Host ""
Write-Host "Ghi chú iOS (.a): Để build cho iOS (aarch64-apple-ios / simulator), hãy chạy ./build_mobile.sh trên macOS có Xcode." -ForegroundColor Cyan
