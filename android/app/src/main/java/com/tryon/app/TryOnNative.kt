package com.tryon.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.net.Uri
import androidx.camera.core.ImageProxy
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

/**
 * Cầu nối JNI trực tiếp tới Rust core (`libtr...` -> `librust_tryon_engine.so`).
 */
object TryOnNative {
    init {
        System.loadLibrary("rust_tryon_engine")
    }

    @JvmStatic
    external fun init(modelPath: String): Long

    @JvmStatic
    external fun processFrameDirect(
        enginePtr: Long,
        frameBuffer: ByteBuffer,
        width: Int,
        height: Int,
        garmentBuffer: ByteBuffer,
        garmentW: Int,
        garmentH: Int,
        targetClassId: Int
    ): Int

    @JvmStatic
    external fun processFrameByteArray(
        enginePtr: Long,
        frameArray: ByteArray,
        width: Int,
        height: Int,
        garmentArray: ByteArray,
        garmentW: Int,
        garmentH: Int,
        targetClassId: Int
    ): Int

    @JvmStatic
    external fun free(enginePtr: Long)
}

/**
 * Danh sách lớp quần áo khớp với `data.yaml` khi train trên Colab (`train_clothes_yolo_colab.ipynb`).
 */
enum class GarmentCategory(val classId: Int, val label: String) {
    SHIRT(0, "Áo thun"),
    PANTS(1, "Quần"),
    JACKET(2, "Áo khoác"),
    DRESS(3, "Váy")
}

/**
 * Quản lý vòng đời TryOnEngine, copy file .onnx từ assets, chuẩn bị buffer RGBA DirectByteBuffer
 * cho ảnh quần áo mẫu và xử lý frame CameraX RGBA_8888.
 */
class TryOnController(private val context: Context) : AutoCloseable {
    companion object {
        const val DEFAULT_MODEL_ASSET = "yolo11n_seg_clothes.onnx"
    }

    private val lock = Any()
    private var enginePtr: Long = 0L

    private var garmentBuffer: ByteBuffer? = null
    private var garmentWidth: Int = 0
    private var garmentHeight: Int = 0

    // Buffer tái sử dụng qua từng frame để hạn chế GC
    private var frameDirectBuffer: ByteBuffer? = null
    private var packedRowBuffer: ByteArray? = null
    private var rawBitmap: Bitmap? = null
    private var uprightBitmap: Bitmap? = null

    var isReady: Boolean = false
        private set

    var initError: String? = null
        private set

    fun initialize(modelAssetName: String = DEFAULT_MODEL_ASSET): Boolean {
        synchronized(lock) {
            if (enginePtr != 0L) return true
            return try {
                val modelFile = ensureAssetCopied(modelAssetName)
                val ptr = TryOnNative.init(modelFile.absolutePath)
                if (ptr == 0L) {
                    initError = "Không thể khởi tạo ONNX session từ ${modelFile.name}"
                    isReady = false
                    false
                } else {
                    enginePtr = ptr
                    initError = null
                    isReady = true
                    true
                }
            } catch (e: Exception) {
                initError = "Chưa có model $modelAssetName trong assets/: ${e.message}"
                isReady = false
                false
            }
        }
    }

    fun setGarmentBitmap(bitmap: Bitmap) {
        val rgbaBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        val byteCount = rgbaBitmap.width * rgbaBitmap.height * 4
        val direct = ByteBuffer.allocateDirect(byteCount)
        rgbaBitmap.copyPixelsToBuffer(direct)
        direct.rewind()

        synchronized(lock) {
            garmentBuffer = direct
            garmentWidth = rgbaBitmap.width
            garmentHeight = rgbaBitmap.height
        }
    }

    fun setGarmentFromUri(uri: Uri): Boolean {
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                val bmp = BitmapFactory.decodeStream(stream) ?: return false
                setGarmentBitmap(bmp)
                true
            } ?: false
        } catch (_: Exception) {
            false
        }
    }

    fun setSampleGarment(category: GarmentCategory, colorHex: Int) {
        // Thử tải ảnh PNG từ assets/garments/ trước; nếu chưa có thì tạo áo mẫu RGBA sắc nét
        val assetPath = "garments/${category.name.lowercase()}.png"
        val loaded = try {
            context.assets.open(assetPath).use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (_: Exception) {
            null
        }
        setGarmentBitmap(loaded ?: createProceduralGarment(category, colorHex))
    }

    /**
     * Nhận ImageProxy định dạng RGBA_8888 từ CameraX, xoay đúng chiều hiển thị,
     * gọi Rust `processFrameDirect` để tách mask và ghép áo mới trực tiếp trên DirectByteBuffer.
     */
    fun processCameraFrame(
        imageProxy: ImageProxy,
        targetClassId: Int,
        mirrorHorizontal: Boolean
    ): Bitmap? {
        val plane = imageProxy.planes.firstOrNull() ?: return null
        val srcW = imageProxy.width
        val srcH = imageProxy.height
        val rowStride = plane.rowStride
        val srcBuffer = plane.buffer
        srcBuffer.rewind()

        val rotation = imageProxy.imageInfo.rotationDegrees

        synchronized(lock) {
            val gBuf = garmentBuffer
            val gw = garmentWidth
            val gh = garmentHeight

            // Fast path zero-copy nếu frame đã thẳng, không padding row và không lật gương
            if (rotation == 0 && !mirrorHorizontal && rowStride == srcW * 4 && srcBuffer.isDirect) {
                if (enginePtr != 0L && gBuf != null && gw > 0 && gh > 0) {
                    TryOnNative.processFrameDirect(
                        enginePtr = enginePtr,
                        frameBuffer = srcBuffer,
                        width = srcW,
                        height = srcH,
                        garmentBuffer = gBuf,
                        garmentW = gw,
                        garmentH = gh,
                        targetClassId = targetClassId
                    )
                    srcBuffer.rewind()
                }
                val outBmp = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)
                outBmp.copyPixelsFromBuffer(srcBuffer)
                return outBmp
            }

            // Copy buffer CameraX vào rawBitmap (xử lý được thiết bị có rowStride > width * 4)
            val raw = ensureRawBitmap(srcW, srcH)
            if (rowStride == srcW * 4) {
                raw.copyPixelsFromBuffer(srcBuffer)
            } else {
                val packed = ensurePackedByteArray(srcW * srcH * 4)
                val rowBytes = srcW * 4
                for (y in 0 until srcH) {
                    srcBuffer.position(y * rowStride)
                    srcBuffer.get(packed, y * rowBytes, rowBytes)
                }
                raw.copyPixelsFromBuffer(ByteBuffer.wrap(packed))
            }

            val outW = if (rotation == 90 || rotation == 270) srcH else srcW
            val outH = if (rotation == 90 || rotation == 270) srcW else srcH
            val upright = ensureUprightBitmap(outW, outH)

            val matrix = Matrix().apply {
                postTranslate(-srcW / 2f, -srcH / 2f)
                if (rotation != 0) postRotate(rotation.toFloat())
                if (mirrorHorizontal) postScale(-1f, 1f)
                postTranslate(outW / 2f, outH / 2f)
            }
            Canvas(upright).drawBitmap(raw, matrix, null)

            val directBuf = ensureFrameDirectBuffer(outW * outH * 4)
            directBuf.rewind()
            upright.copyPixelsToBuffer(directBuf)
            directBuf.rewind()

            if (enginePtr != 0L && gBuf != null && gw > 0 && gh > 0) {
                TryOnNative.processFrameDirect(
                    enginePtr = enginePtr,
                    frameBuffer = directBuf,
                    width = outW,
                    height = outH,
                    garmentBuffer = gBuf,
                    garmentW = gw,
                    garmentH = gh,
                    targetClassId = targetClassId
                )
                directBuf.rewind()
            }

            val resultBitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
            resultBitmap.copyPixelsFromBuffer(directBuf)
            return resultBitmap
        }
    }

    override fun close() {
        synchronized(lock) {
            if (enginePtr != 0L) {
                TryOnNative.free(enginePtr)
                enginePtr = 0L
            }
            isReady = false
            rawBitmap?.recycle()
            rawBitmap = null
            uprightBitmap?.recycle()
            uprightBitmap = null
        }
    }

    private fun ensureAssetCopied(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (!outFile.exists() || outFile.length() == 0L) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile
    }

    private fun ensureFrameDirectBuffer(byteCount: Int): ByteBuffer {
        val existing = frameDirectBuffer
        if (existing != null && existing.capacity() >= byteCount) {
            existing.clear()
            existing.limit(byteCount)
            return existing
        }
        return ByteBuffer.allocateDirect(byteCount).also { frameDirectBuffer = it }
    }

    private fun ensurePackedByteArray(byteCount: Int): ByteArray {
        val existing = packedRowBuffer
        if (existing != null && existing.size >= byteCount) return existing
        return ByteArray(byteCount).also { packedRowBuffer = it }
    }

    private fun ensureRawBitmap(w: Int, h: Int): Bitmap {
        val existing = rawBitmap
        if (existing != null && existing.width == w && existing.height == h) return existing
        existing?.recycle()
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { rawBitmap = it }
    }

    private fun ensureUprightBitmap(w: Int, h: Int): Bitmap {
        val existing = uprightBitmap
        if (existing != null && existing.width == w && existing.height == h) return existing
        existing?.recycle()
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { uprightBitmap = it }
    }

    private fun createProceduralGarment(category: GarmentCategory, colorHex: Int): Bitmap {
        val w = 256
        val h = 256
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.TRANSPARENT)
        val canvas = Canvas(bmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = colorHex
            style = Paint.Style.FILL
        }
        val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.argb(120, 255, 255, 255)
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }

        val path = Path()
        when (category) {
            GarmentCategory.SHIRT, GarmentCategory.JACKET -> {
                path.moveTo(64f, 24f)
                path.lineTo(100f, 16f)
                path.quadTo(128f, 44f, 156f, 16f)
                path.lineTo(192f, 24f)
                path.lineTo(244f, 88f)
                path.lineTo(208f, 118f)
                path.lineTo(192f, 92f)
                path.lineTo(192f, 240f)
                path.lineTo(64f, 240f)
                path.lineTo(64f, 92f)
                path.lineTo(48f, 118f)
                path.lineTo(12f, 88f)
                path.close()
            }
            GarmentCategory.PANTS -> {
                path.moveTo(56f, 16f)
                path.lineTo(200f, 16f)
                path.lineTo(212f, 244f)
                path.lineTo(148f, 244f)
                path.lineTo(128f, 108f)
                path.lineTo(108f, 244f)
                path.lineTo(44f, 244f)
                path.close()
            }
            GarmentCategory.DRESS -> {
                path.moveTo(84f, 16f)
                path.quadTo(128f, 42f, 172f, 16f)
                path.lineTo(180f, 86f)
                path.lineTo(232f, 244f)
                path.lineTo(24f, 244f)
                path.lineTo(76f, 86f)
                path.close()
            }
        }
        canvas.drawPath(path, paint)
        canvas.drawPath(path, stroke)
        return bmp
    }
}
