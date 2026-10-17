package com.macewan.hackathon

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.Executors
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import android.graphics.RectF


@Composable
fun CameraScreen() {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    if (hasPermission) {
        CameraPreview()
    } else {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera permission needed")
        }
        LaunchedEffect(Unit) { launcher.launch(Manifest.permission.CAMERA) }
    }
}

private fun createLandmarker(
    context: Context,
    onResult: (List<Offset>) -> Unit
): HandLandmarker {
    val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
    val options = HandLandmarker.HandLandmarkerOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .setNumHands(1)
        .setResultListener { result, _ ->
            val hand = result.landmarks().firstOrNull()
            onResult(hand?.map { Offset(it.x(), it.y()) } ?: emptyList())
        }
        .setErrorListener { e -> Log.e("HandTracker", "MediaPipe error", e) }
        .build()
    return HandLandmarker.createFromOptions(context, options)
}
// create the object points for the object in view
data class DetectedItem(
    val label: String,
    val score: Float,
    val boundingBox: RectF //coords
)

private fun createObjectDetector(
    context: Context,
    onResult: (List<DetectedItem>) -> Unit
): ObjectDetector {
    val base = BaseOptions.builder()
        .setModelAssetPath("efficientdet_lite0.tflite")
        .build()

    val options = ObjectDetector.ObjectDetectorOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .setMaxResults(5) // Only keep top 5 objects on table
        .setScoreThreshold(0.45f) // Ignore low confidence predictions
        .setResultListener { result: ObjectDetectorResult, _ ->
            val items = result.detections().mapNotNull { detection ->
                val category = detection.categories().firstOrNull() ?: return@mapNotNull null
                val box = detection.boundingBox()
                DetectedItem(
                    label = category.categoryName(),
                    score = category.score(),
                    boundingBox = RectF(box.left, box.top, box.right, box.bottom)
                )
            }
            onResult(items)
        }
        .setErrorListener { e -> Log.e("ObjectDetector", "MediaPipe Error", e) }
        .build()

    return ObjectDetector.createFromOptions(context, options)
}

private val HAND_LINKS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,
    0 to 5, 5 to 6, 6 to 7, 7 to 8,
    5 to 9, 9 to 10, 10 to 11, 11 to 12,
    9 to 13, 13 to 14, 14 to 15, 15 to 16,
    13 to 17, 17 to 18, 18 to 19, 19 to 20, 0 to 17
)

@Composable
fun CameraPreview() {
    val context = LocalContext.current
    val lifecycleOwner = context as ComponentActivity

    var points by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var fps by remember { mutableIntStateOf(0) }

    var detectedObjects by remember { mutableStateOf<List<DetectedItem>>(emptyList()) }
    val objectDetector = remember { createObjectDetector(context) { detectedObjects = it } }

    val executor = remember { Executors.newSingleThreadExecutor() }
    val landmarker = remember { createLandmarker(context) { points = it } }
    val textMeasurer = rememberTextMeasurer()

    DisposableEffect(Unit) {
        var lastTimestamp = 0L

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frames = 0
            var windowStart = SystemClock.uptimeMillis()

            analysis.setAnalyzer(executor) { proxy ->
                try {
                    val raw = proxy.toBitmap()
                    val matrix = Matrix().apply {
                        postRotate(proxy.imageInfo.rotationDegrees.toFloat())
                    }
                    val upright = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
                    frame = upright

                    val mpImage = BitmapImageBuilder(upright).build()

                    // each timestamp has to be greater than the last
                    var currentTimestamp = SystemClock.uptimeMillis()
                    if (currentTimestamp <= lastTimestamp) {
                        currentTimestamp = lastTimestamp + 1
                    }
                    lastTimestamp = currentTimestamp

                    // send the good stamps to the detector
                    landmarker.detectAsync(mpImage, currentTimestamp)
                    objectDetector.detectAsync(mpImage, currentTimestamp)

                } catch (e: Exception) {
                    Log.e("CameraPreview", "Frame processing error", e)
                } finally {
                    // close the deliverance of frames
                    proxy.close()
                }

                frames++
                val now = SystemClock.uptimeMillis()
                if (now - windowStart >= 1000) {
                    fps = frames
                    frames = 0
                    windowStart = now
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            executor.shutdown()
            landmarker.close()
            objectDetector.close()
        }
    }

    Box(
        Modifier.fillMaxSize().background(Color.Black).statusBarsPadding(),
        contentAlignment = Alignment.TopCenter
    ) {
        val bmp = frame
        if (bmp != null) {
            // Image and overlay share one box with the bitmap's own aspect ratio
            Box(Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height)) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )
                Canvas(Modifier.fillMaxSize()) {
                    if (points.size == 21) {
                        HAND_LINKS.forEach { (a, b) ->
                            drawLine(
                                Color.Green,
                                Offset(points[a].x * size.width, points[a].y * size.height),
                                Offset(points[b].x * size.width, points[b].y * size.height),
                                strokeWidth = 6f
                            )
                        }
                        points.forEach { p ->
                            drawCircle(Color.Red, 10f, Offset(p.x * size.width, p.y * size.height))
                        }
                    }

                    // draw the boxes for objects
                    detectedObjects.forEach { obj ->
                        // mediapipe objectdetector returns pixel coordinates relative to the upright bitmap size
                        // scale them to match the canvas size
                        val scaleX = size.width / bmp.width
                        val scaleY = size.height / bmp.height

                        //coords for top left
                        val left = obj.boundingBox.left * scaleX
                        val top = obj.boundingBox.top * scaleY
                        val boxWidth = obj.boundingBox.width() * scaleX
                        val boxHeight = obj.boundingBox.height() * scaleY

                        //draw the rectangle inside the bounding box limits
                        drawRect(
                            color = Color.Yellow,
                            topLeft = Offset(left, top),
                            size = Size(boxWidth, boxHeight),
                            style = Stroke(width = 4f)
                        )
                        //label format
                        val textLabel = "${obj.label} ${(obj.score * 100).toInt()}%"

                        //measure the size of text for background
                        val textLayoutResult = textMeasurer.measure(
                            text = textLabel,
                            style = TextStyle(
                                color = Color.Yellow,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold
                            )
                        )

                        // Position text above the box (or just inside if near top edge of screen)
                        val textY = if (top > textLayoutResult.size.height + 10) {
                            top - textLayoutResult.size.height - 8
                        } else {
                            top + 8
                        }

                        // 3. Draw a dark backdrop behind text so it's readable
                        drawRect(
                            color = Color.Black.copy(alpha = 0.75f),
                            topLeft = Offset(left, textY),
                            size = Size(
                                width = textLayoutResult.size.width + 16f,
                                height = textLayoutResult.size.height + 8f
                            )
                        )

                        drawText(textLayoutResult = textLayoutResult, topLeft = Offset(left + 8f, textY + 4f))
                    }
                }
            }
        }
        Text(
            "FPS: $fps",
            color = Color.White,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(16.dp)
        )
    }
}