package com.tryon.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

private data class GarmentSwatch(val name: String, val colorHex: Int)

private val SAMPLE_SWATCHES = listOf(
    GarmentSwatch("Đỏ Ruby", 0xFFE53935.toInt()),
    GarmentSwatch("Xanh Navy", 0xFF1E3A8A.toInt()),
    GarmentSwatch("Xanh Rêu", 0xFF2E7D32.toInt()),
    GarmentSwatch("Vàng Nghệ", 0xFFF9A825.toInt()),
    GarmentSwatch("Tím Pastel", 0xFF7E57C2.toInt()),
    GarmentSwatch("Đen Carbon", 0xFF212121.toInt())
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize()) {
                    TryOnCameraScreen()
                }
            }
        }
    }
}

@Composable
fun TryOnCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    val controller = remember { TryOnController(context) }
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    var selectedCategory by remember { mutableStateOf(GarmentCategory.SHIRT) }
    var selectedSwatch by remember { mutableStateOf(SAMPLE_SWATCHES.first()) }
    var useFrontCamera by remember { mutableStateOf(true) }
    var modelStatus by remember { mutableStateOf<String?>(null) }
    var fps by remember { mutableIntStateOf(0) }
    var currentFrame by remember { mutableStateOf<Bitmap?>(null) }

    val activeCategoryState = rememberUpdatedState(selectedCategory)
    val activeFrontCamState = rememberUpdatedState(useFrontCamera)

    val galleryPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            controller.setGarmentFromUri(uri)
        }
    }

    LaunchedEffect(Unit) {
        val ok = controller.initialize()
        modelStatus = if (ok) null else controller.initError
        controller.setSampleGarment(selectedCategory, selectedSwatch.colorHex)
    }

    LaunchedEffect(selectedCategory, selectedSwatch) {
        controller.setSampleGarment(selectedCategory, selectedSwatch.colorHex)
    }

    DisposableEffect(Unit) {
        onDispose {
            controller.close()
            cameraExecutor.shutdown()
        }
    }

    if (!hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("Cần quyền Camera để thử quần áo real-time", color = Color.White)
                Spacer(modifier = Modifier.height(12.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Cấp quyền Camera")
                }
            }
        }
        return
    }

    // Khởi tạo & bind CameraX ImageAnalysis (RGBA_8888, KEEP_ONLY_LATEST)
    DisposableEffect(useFrontCamera, hasCameraPermission) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val listener = Runnable {
            val cameraProvider = cameraProviderFuture.get()
            val resolutionSelector = ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(resolutionSelector)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()
            val mainExecutor = ContextCompat.getMainExecutor(context)

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val processed = controller.processCameraFrame(
                        imageProxy = imageProxy,
                        targetClassId = activeCategoryState.value.classId,
                        mirrorHorizontal = activeFrontCamState.value
                    )
                    frameCounter++
                    val now = System.currentTimeMillis()
                    var newFps: Int? = null
                    if (now - lastFpsTimestamp >= 1000L) {
                        newFps = frameCounter
                        frameCounter = 0
                        lastFpsTimestamp = now
                    }
                    if (processed != null || newFps != null) {
                        mainExecutor.execute {
                            if (processed != null) currentFrame = processed
                            if (newFps != null) fps = newFps
                        }
                    }
                } finally {
                    imageProxy.close()
                }
            }

            val selector = if (useFrontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(lifecycleOwner, selector, imageAnalysis)
        }
        cameraProviderFuture.addListener(listener, ContextCompat.getMainExecutor(context))

        onDispose {
            runCatching { cameraProviderFuture.get().unbindAll() }
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        currentFrame?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "Real-time Virtual Try-On Camera",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }

        // Thanh trạng thái phía trên: FPS + Đổi camera + Cảnh báo nếu thiếu .onnx
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    color = Color.Black.copy(alpha = 0.6f),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text(
                        text = "Rust AI • $fps FPS",
                        color = Color(0xFF4ADE80),
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }

                FilledTonalButton(onClick = { useFrontCamera = !useFrontCamera }) {
                    Text(if (useFrontCamera) "Cam Sau" else "Cam Trước")
                }
            }

            if (modelStatus != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = Color(0xFFB91C1C).copy(alpha = 0.85f),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = modelStatus ?: "",
                        color = Color.White,
                        fontSize = 12.sp,
                        modifier = Modifier.padding(10.dp)
                    )
                }
            }
        }

        // Bảng điều khiển phía dưới: chọn loại quần áo, màu mẫu hoặc ảnh PNG từ máy
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .background(Color.Black.copy(alpha = 0.65f))
                .navigationBarsPadding()
                .padding(14.dp)
        ) {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(GarmentCategory.entries) { category ->
                    FilterChip(
                        selected = selectedCategory == category,
                        onClick = { selectedCategory = category },
                        label = { Text(category.label) }
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                LazyRow(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(SAMPLE_SWATCHES) { swatch ->
                        val isSelected = selectedSwatch == swatch
                        Box(
                            modifier = Modifier
                                .size(38.dp)
                                .clip(CircleShape)
                                .background(Color(swatch.colorHex))
                                .border(
                                    width = if (isSelected) 3.dp else 1.dp,
                                    color = if (isSelected) Color.White else Color.Gray,
                                    shape = CircleShape
                                )
                                .clickable { selectedSwatch = swatch }
                        )
                    }
                }

                Spacer(modifier = Modifier.width(10.dp))

                Button(onClick = { galleryPicker.launch("image/*") }) {
                    Text("Chọn PNG")
                }
            }
        }
    }
}
