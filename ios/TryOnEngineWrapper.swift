import Accelerate
import CoreVideo
import Foundation
import UIKit

/// Danh sách lớp quần áo khớp với `data.yaml` khi train trên Google Colab.
enum GarmentCategory: Int, CaseIterable, Identifiable {
    case shirt = 0
    case pants = 1
    case jacket = 2
    case dress = 3

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .shirt: return "Áo thun"
        case .pants: return "Quần"
        case .jacket: return "Áo khoác"
        case .dress: return "Váy"
        }
    }
}

/// Wrapper Swift quản lý con trỏ OpaquePointer tới Rust TryOnEngine (C-ABI FFI).
final class TryOnEngineWrapper {
    private let lock = NSLock()
    private var enginePtr: OpaquePointer?

    private var garmentRGBA: [UInt8] = []
    private var garmentWidth: UInt32 = 0
    private var garmentHeight: UInt32 = 0
    private var rgbaScratchBuffer: [UInt8] = []

    var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return enginePtr != nil
    }

    init?(modelPath: String) {
        guard let ptr = modelPath.withCString({ tryon_init($0) }) else {
            return nil
        }
        self.enginePtr = ptr
    }

    deinit {
        lock.lock()
        if let ptr = enginePtr {
            tryon_free(ptr)
            enginePtr = nil
        }
        lock.unlock()
    }

    /// Chuyển UIImage áo/quần mẫu sang buffer RGBA8888 unpremultiplied để Rust blend lên camera.
    func setGarmentImage(_ image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return }

        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        bytes.withUnsafeMutableBytes { rawPtr in
            guard let baseAddress = rawPtr.baseAddress,
                  let ctx = CGContext(
                      data: baseAddress,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: width * 4,
                      space: colorSpace,
                      bitmapInfo: bitmapInfo
                  )
            else { return }
            ctx.clear(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        // Unpremultiply RGB để giữ viền áo sắc nét khi Rust blend theo alpha
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let a = Int(bytes[i + 3])
            if a > 0 && a < 255 {
                bytes[i] = UInt8(min(255, Int(bytes[i]) * 255 / a))
                bytes[i + 1] = UInt8(min(255, Int(bytes[i + 1]) * 255 / a))
                bytes[i + 2] = UInt8(min(255, Int(bytes[i + 2]) * 255 / a))
            }
        }

        lock.lock()
        garmentRGBA = bytes
        garmentWidth = UInt32(width)
        garmentHeight = UInt32(height)
        lock.unlock()
    }

    /// Tạo áo mẫu RGBA mặc định theo loại trang phục và màu sắc khi chưa chọn ảnh từ thư viện.
    static func makeSampleGarment(category: GarmentCategory, color: UIColor) -> UIImage {
        let size = CGSize(width: 256, height: 256)
        let format = UIGraphicsImageRendererFormat.default()
        format.opaque = false
        format.scale = 1.0

        return UIGraphicsImageRenderer(size: size, format: format).image { rendererCtx in
            let ctx = rendererCtx.cgContext
            ctx.clear(CGRect(origin: .zero, size: size))

            let path = UIBezierPath()
            switch category {
            case .shirt, .jacket:
                path.move(to: CGPoint(x: 64, y: 24))
                path.addLine(to: CGPoint(x: 100, y: 16))
                path.addQuadCurve(to: CGPoint(x: 156, y: 16), controlPoint: CGPoint(x: 128, y: 44))
                path.addLine(to: CGPoint(x: 192, y: 24))
                path.addLine(to: CGPoint(x: 244, y: 88))
                path.addLine(to: CGPoint(x: 208, y: 118))
                path.addLine(to: CGPoint(x: 192, y: 92))
                path.addLine(to: CGPoint(x: 192, y: 240))
                path.addLine(to: CGPoint(x: 64, y: 240))
                path.addLine(to: CGPoint(x: 64, y: 92))
                path.addLine(to: CGPoint(x: 48, y: 118))
                path.addLine(to: CGPoint(x: 12, y: 88))
                path.close()
            case .pants:
                path.move(to: CGPoint(x: 56, y: 16))
                path.addLine(to: CGPoint(x: 200, y: 16))
                path.addLine(to: CGPoint(x: 212, y: 244))
                path.addLine(to: CGPoint(x: 148, y: 244))
                path.addLine(to: CGPoint(x: 128, y: 108))
                path.addLine(to: CGPoint(x: 108, y: 244))
                path.addLine(to: CGPoint(x: 44, y: 244))
                path.close()
            case .dress:
                path.move(to: CGPoint(x: 84, y: 16))
                path.addQuadCurve(to: CGPoint(x: 172, y: 16), controlPoint: CGPoint(x: 128, y: 42))
                path.addLine(to: CGPoint(x: 180, y: 86))
                path.addLine(to: CGPoint(x: 232, y: 244))
                path.addLine(to: CGPoint(x: 24, y: 244))
                path.addLine(to: CGPoint(x: 76, y: 86))
                path.close()
            }

            color.setFill()
            path.fill()

            UIColor.white.withAlphaComponent(0.45).setStroke()
            path.lineWidth = 4
            path.stroke()
        }
    }

    /// Xử lý CVPixelBuffer (kCVPixelFormatType_32BGRA) -> chuyển RGBA bằng Accelerate -> gọi Rust -> trả CGImage.
    func processPixelBuffer(_ pixelBuffer: CVPixelBuffer, targetClassId: Int) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dstBytesPerRow = width * 4
        let requiredBytes = dstBytesPerRow * height
        guard width > 0, height > 0 else { return nil }

        lock.lock()
        defer { lock.unlock() }

        if rgbaScratchBuffer.count != requiredBytes {
            rgbaScratchBuffer = [UInt8](repeating: 0, count: requiredBytes)
        }

        // Chuyển BGRA (AVFoundation) -> RGBA (Rust TryOnEngine) bằng vImage SIMD
        rgbaScratchBuffer.withUnsafeMutableBytes { dstRaw in
            guard let dstBase = dstRaw.baseAddress else { return }
            var srcBuf = vImage_Buffer(
                data: srcBase,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: srcBytesPerRow
            )
            var dstBuf = vImage_Buffer(
                data: dstBase,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: dstBytesPerRow
            )
            var permuteMap: [UInt8] = [2, 1, 0, 3]
            vImagePermuteChannels_ARGB8888(&srcBuf, &dstBuf, &permuteMap, vImage_Flags(kvImageNoFlags))
        }

        if let ptr = enginePtr, !garmentRGBA.isEmpty, garmentWidth > 0, garmentHeight > 0 {
            let gw = garmentWidth
            let gh = garmentHeight
            rgbaScratchBuffer.withUnsafeMutableBufferPointer { framePtr in
                garmentRGBA.withUnsafeBufferPointer { gPtr in
                    guard let fBase = framePtr.baseAddress, let gBase = gPtr.baseAddress else {
                        return
                    }
                    _ = tryon_process_frame(
                        ptr,
                        fBase,
                        UInt32(width),
                        UInt32(height),
                        gBase,
                        gw,
                        gh,
                        max(0, targetClassId)
                    )
                }
            }
        }

        let outputData = Data(rgbaScratchBuffer)
        guard let provider = CGDataProvider(data: outputData as CFData) else {
            return nil
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: dstBytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}
