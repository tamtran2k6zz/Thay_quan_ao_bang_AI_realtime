import AVFoundation
import SwiftUI
import UIKit

private struct GarmentSwatchItem: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let color: UIColor
}

private let sampleSwatches: [GarmentSwatchItem] = [
    GarmentSwatchItem(name: "Đỏ Ruby", color: UIColor(red: 0.90, green: 0.22, blue: 0.21, alpha: 1.0)),
    GarmentSwatchItem(name: "Xanh Navy", color: UIColor(red: 0.12, green: 0.23, blue: 0.54, alpha: 1.0)),
    GarmentSwatchItem(name: "Xanh Rêu", color: UIColor(red: 0.18, green: 0.49, blue: 0.20, alpha: 1.0)),
    GarmentSwatchItem(name: "Vàng Nghệ", color: UIColor(red: 0.98, green: 0.66, blue: 0.15, alpha: 1.0)),
    GarmentSwatchItem(name: "Tím Pastel", color: UIColor(red: 0.49, green: 0.34, blue: 0.76, alpha: 1.0)),
    GarmentSwatchItem(name: "Đen Carbon", color: UIColor(red: 0.13, green: 0.13, blue: 0.13, alpha: 1.0))
]

final class CameraTryOnManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var currentFrame: CGImage?
    @Published var fps: Int = 0
    @Published var modelWarning: String?
    @Published var permissionDenied: Bool = false
    @Published var isFrontCamera: Bool = true

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "com.tryon.ios.camera.queue", qos: .userInteractive)

    private var wrapper: TryOnEngineWrapper?
    private var activeCategory: GarmentCategory = .shirt
    private var frameCount = 0
    private var lastFpsTime = CFAbsoluteTimeGetCurrent()

    override init() {
        super.init()
        loadEngine()
    }

    private func loadEngine() {
        if let modelPath = Bundle.main.path(forResource: "yolo11n_seg_clothes", ofType: "onnx"),
           let engine = TryOnEngineWrapper(modelPath: modelPath) {
            self.wrapper = engine
            self.modelWarning = nil
        } else {
            self.wrapper = nil
            self.modelWarning = "Chưa tìm thấy yolo11n_seg_clothes.onnx trong App Bundle (đang chạy camera preview)."
        }
    }

    func updateGarment(category: GarmentCategory, color: UIColor) {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.activeCategory = category
            let sample = TryOnEngineWrapper.makeSampleGarment(category: category, color: color)
            self.wrapper?.setGarmentImage(sample)
        }
    }

    func updateCustomGarment(image: UIImage) {
        processingQueue.async { [weak self] in
            self?.wrapper?.setGarmentImage(image)
        }
    }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureSession(front: isFrontCamera)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.configureSession(front: self?.isFrontCamera ?? true)
                    } else {
                        self?.permissionDenied = true
                    }
                }
            }
        default:
            DispatchQueue.main.async {
                self.permissionDenied = true
            }
        }
    }

    func stop() {
        processingQueue.async { [weak self] in
            if self?.session.isRunning == true {
                self?.session.stopRunning()
            }
        }
    }

    func toggleCamera() {
        let nextFront = !isFrontCamera
        isFrontCamera = nextFront
        configureSession(front: nextFront)
    }

    private func configureSession(front: Bool) {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .vga640x480

            for input in self.session.inputs {
                self.session.removeInput(input)
            }

            let position: AVCaptureDevice.Position = front ? .front : .back
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input)
            else {
                self.session.commitConfiguration()
                return
            }
            self.session.addInput(input)

            if !self.session.outputs.contains(self.videoOutput) {
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
                ]
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.setSampleBufferDelegate(self, queue: self.processingQueue)
                if self.session.canAddOutput(self.videoOutput) {
                    self.session.addOutput(self.videoOutput)
                }
            }

            if let connection = self.videoOutput.connection(with: .video) {
                if #available(iOS 17.0, *) {
                    if connection.isVideoRotationAngleSupported(90) {
                        connection.videoRotationAngle = 90
                    }
                } else if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
                if connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = front
                }
            }

            self.session.commitConfiguration()
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let processed = wrapper?.processPixelBuffer(pixelBuffer, targetClassId: activeCategory.rawValue)

        frameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        var computedFps: Int?
        if now - lastFpsTime >= 1.0 {
            computedFps = frameCount
            frameCount = 0
            lastFpsTime = now
        }

        if let cgImage = processed {
            DispatchQueue.main.async {
                self.currentFrame = cgImage
                if let newFps = computedFps {
                    self.fps = newFps
                }
            }
        } else if let cgImage = fallbackCGImage(from: pixelBuffer) {
            DispatchQueue.main.async {
                self.currentFrame = cgImage
                if let newFps = computedFps {
                    self.fps = newFps
                }
            }
        }
    }

    private func fallbackCGImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )
        guard let ctx = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }
        return ctx.makeImage()
    }
}

struct CameraTryOnView: View {
    @StateObject private var manager = CameraTryOnManager()
    @State private var selectedCategory: GarmentCategory = .shirt
    @State private var selectedSwatch: GarmentSwatchItem = sampleSwatches[0]
    @State private var showPhotoPicker = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if manager.permissionDenied {
                VStack(spacing: 12) {
                    Text("Cần quyền Camera để thử quần áo real-time")
                        .foregroundColor(.white)
                    Button("Mở Cài đặt") {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else if let frame = manager.currentFrame {
                Image(decorative: frame, scale: 1.0)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else {
                ProgressView("Đang khởi động camera...")
                    .tint(.white)
                    .foregroundColor(.white)
            }

            VStack {
                HStack {
                    Text("Rust AI • \(manager.fps) FPS")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(Color(red: 0.29, green: 0.87, blue: 0.50))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.6))
                        .clipShape(Capsule())

                    Spacer()

                    Button(action: { manager.toggleCamera() }) {
                        Text(manager.isFrontCamera ? "Cam Sau" : "Cam Trước")
                            .font(.system(size: 13, weight: .semibold))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Color.white.opacity(0.2))
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)

                if let warning = manager.modelWarning {
                    Text(warning)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(10)
                        .padding(.horizontal, 16)
                }

                Spacer()

                VStack(spacing: 12) {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(GarmentCategory.allCases) { category in
                                Button(action: {
                                    selectedCategory = category
                                    manager.updateGarment(category: category, color: selectedSwatch.color)
                                }) {
                                    Text(category.label)
                                        .font(.system(size: 13, weight: .medium))
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 8)
                                        .background(
                                            selectedCategory == category
                                                ? Color.white
                                                : Color.white.opacity(0.18)
                                        )
                                        .foregroundColor(selectedCategory == category ? .black : .white)
                                        .clipShape(Capsule())
                                }
                            }
                        }
                    }

                    HStack {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 10) {
                                ForEach(sampleSwatches) { swatch in
                                    Circle()
                                        .fill(Color(swatch.color))
                                        .frame(width: 34, height: 34)
                                        .overlay(
                                            Circle().stroke(
                                                selectedSwatch == swatch ? Color.white : Color.gray,
                                                lineWidth: selectedSwatch == swatch ? 3 : 1
                                            )
                                        )
                                        .onTapGesture {
                                            selectedSwatch = swatch
                                            manager.updateGarment(
                                                category: selectedCategory,
                                                color: swatch.color
                                            )
                                        }
                                }
                            }
                        }

                        Button("Chọn PNG") {
                            showPhotoPicker = true
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.65))
            }
        }
        .onAppear {
            manager.updateGarment(category: selectedCategory, color: selectedSwatch.color)
            manager.start()
        }
        .onDisappear {
            manager.stop()
        }
        .sheet(isPresented: $showPhotoPicker) {
            GarmentPhotoPicker { image in
                if let image = image {
                    manager.updateCustomGarment(image: image)
                }
            }
        }
    }
}

private struct GarmentPhotoPicker: UIViewControllerRepresentable {
    var onPick: (UIImage?) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        private let onPick: (UIImage?) -> Void

        init(onPick: @escaping (UIImage?) -> Void) {
            self.onPick = onPick
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            let image = info[.originalImage] as? UIImage
            picker.dismiss(animated: true) {
                self.onPick(image)
            }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true) {
                self.onPick(nil)
            }
        }
    }
}
