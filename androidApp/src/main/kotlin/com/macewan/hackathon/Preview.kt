package com.macewan.hackathon

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
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
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlinx.coroutines.delay
import androidx.camera.core.Camera
import android.speech.tts.TextToSpeech
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

enum class AppMode {
    IDLE, LISTING, TARGET_RANGE
}

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

private fun createLandmarker(context: Context, onResult: (List<Offset>) -> Unit): HandLandmarker {
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

data class DetectedItem(val label: String, val score: Float, val boundingBox: RectF)
val IGNORED_LABELS = setOf("person", "dining table")
/*private fun createObjectDetector(context: Context, onResult: (List<DetectedItem>) -> Unit): ObjectDetector {
    val base = BaseOptions.builder()
        .setModelAssetPath("efficientdet_lite0.tflite")
        .build()

    val options = ObjectDetector.ObjectDetectorOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .setMaxResults(10)
        .setScoreThreshold(0.45f)
        .setResultListener { result: ObjectDetectorResult, _ ->
            val items = result.detections().mapNotNull { detection ->
                val category = detection.categories().firstOrNull() ?: return@mapNotNull null
                val label = category.categoryName().lowercase()
                if (label in IGNORED_LABELS ){
                    return@mapNotNull null
                }
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
*/

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
    val alignment = rememberCameraAlignment()   // teammate's tilt code

    var points by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var fps by remember { mutableIntStateOf(0) }

    // camera and torch states
    var camera by remember { mutableStateOf<Camera?>(null) }
    var isTorchOn by remember { mutableStateOf(false) }

    // tts
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }

    var detectedObjects by remember { mutableStateOf<List<DetectedItem>>(emptyList()) }
    val yolo = remember { YoloDetector(context) }
    val yoloExecutor = remember { Executors.newSingleThreadExecutor() }
    val yoloBusy = remember { AtomicBoolean(false) }

    val executor = remember { Executors.newSingleThreadExecutor() }
    val landmarker = remember { createLandmarker(context) { points = it } }
    val textMeasurer = rememberTextMeasurer()

    val beeper = remember { ProximityBeeper() }
    LaunchedEffect(Unit) { beeper.start(this) }
    val speaker = remember { Speaker(context) }
    rememberShakeDetector(enabled = true) {
        isTorchOn = !isTorchOn
        speaker.say(if (isTorchOn) "Flashlight on" else "Flashlight off")
    }
    // --- State for Tap-Driven Flow ---
    var appMode by remember { mutableStateOf(AppMode.IDLE) }
    var scanning by remember { mutableStateOf(false) }
    var listening by remember { mutableStateOf(false) }
    var targetLabel by remember { mutableStateOf<String?>(null) }
    var lockedBox by remember { mutableStateOf<RectF?>(null) }
    var lockedAt by remember { mutableLongStateOf(0L) }
    var lastSpoken by remember { mutableStateOf("Tap screen to list items") }

    // NEW: Track if we've already announced finding the target so it doesn't spam the speaker
    var hasAnnouncedFound by remember { mutableStateOf(false) }

    // --- Voice Setup ---
    var micOk by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { micOk = it }

    LaunchedEffect(camera, isTorchOn) {
        val cam = camera ?: return@LaunchedEffect
        if (cam.cameraInfo.hasFlashUnit()) {
            cam.cameraControl.enableTorch(isTorchOn)
        }
    }
    LaunchedEffect(Unit) {
        if (!micOk) micLauncher.launch(Manifest.permission.RECORD_AUDIO)

        delay(1000)
        speaker.say("Tap the screen to list items")

        while (true) {
            delay(5000)
            if (appMode == AppMode.IDLE && targetLabel == null) {
                speaker.say("Tap the screen to list items")
            }
        }
    }

    val voice = remember {
        VoiceTarget(context) { label ->
            listening = false
            if (label != null) {
                targetLabel = label
                appMode = AppMode.TARGET_RANGE
                hasAnnouncedFound = false // Reset the announcement tracker
                speaker.say("Looking for $label.")
                lastSpoken = "Target: $label"
            } else {
                speaker.say("I didn't catch that. Tap the screen to scan again.")
                appMode = AppMode.IDLE
                lastSpoken = "Voice failed. Tap to retry."
            }

        }
    }

    // ---- Mode 1: Listing Mode (Triggered by Tap) ----
    LaunchedEffect(appMode) {
        if (appMode == AppMode.LISTING) {
            beeper.distance = null
            scanning = true
            listening = false
            hasAnnouncedFound = false
            lastSpoken = "Scanning..."

            val seen = mutableMapOf<String, Int>()
            val samples = 20
            repeat(samples) {
                detectedObjects.groupBy { it.label }.forEach { (label, _) ->
                    seen[label] = (seen[label] ?: 0) + 1
                }
                delay(100)
            }
            scanning = false

            val stable = seen.filter { it.value >= samples * 0.3 }.keys.toList()

            if (stable.isNotEmpty()) {
                val msg = "I see " + stable.joinToString(", ") + ". Say the item you want."
                speaker.say(msg)
                lastSpoken = "I see: ${stable.joinToString(", ")}"

                delay((msg.length * 80L) + 1000L)

                if (micOk && appMode == AppMode.LISTING) {
                    listening = true
                    lastSpoken = "Listening for your choice..."
                    voice.listen()
                }
            } else {
                val msg = "I don't see anything clearly. Tap to try again."
                speaker.say(msg)
                lastSpoken = "Nothing found."
                appMode = AppMode.IDLE
            }
        }
    }

    // ---- Mode 2: Target Range Proximity Calculations ----
    LaunchedEffect(points, detectedObjects, appMode) {
        val bmp = frame ?: return@LaunchedEffect
        if (appMode != AppMode.TARGET_RANGE || listening || targetLabel == null) {
            beeper.distance = null
            return@LaunchedEffect
        }

        val pts = points
        val now = SystemClock.uptimeMillis()

        val match = detectedObjects.filter { it.label == targetLabel }.maxByOrNull { it.score }
        if (match != null) {
            lockedBox = match.boundingBox
            lockedAt = now

            // NEW: If we just found it and haven't announced it yet, say it!
            if (!hasAnnouncedFound) {
                speaker.say("Found $targetLabel in view")
                hasAnnouncedFound = true
            }

        } else if (now - lockedAt > 3000) {
            lockedBox = null
            // NEW: Reset if the object is out of view for 3 seconds so it will announce again when refound
            hasAnnouncedFound = false
        }

        val box = lockedBox
        var result: Float? = null

        if (pts.size == 21 && box != null) {
            val cx = (box.left + box.right) / 2f
            val cy = (box.top + box.bottom) / 2f
            val maxDist = hypot(bmp.width.toFloat(), bmp.height.toFloat()) * 0.5f

            var best = Float.MAX_VALUE
            for (p in pts) {
                val px = p.x * bmp.width
                val py = p.y * bmp.height
                val d = hypot(px - cx, py - cy)
                if (d < best) best = d
            }
            result = (best / maxDist).coerceIn(0f, 1f)
        }
        beeper.distance = result
    }

    // ---- Camera + Detection Pipeline ----
    DisposableEffect(Unit) {
        var lastTimestamp = 0L
        var cameraProvider: ProcessCameraProvider? = null

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider

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

                    var currentTimestamp = SystemClock.uptimeMillis()
                    if (currentTimestamp <= lastTimestamp) {
                        currentTimestamp = lastTimestamp + 1
                    }
                    lastTimestamp = currentTimestamp

                    landmarker.detectAsync(mpImage, currentTimestamp)
                    if (yoloBusy.compareAndSet(false, true)) {
                        yoloExecutor.execute {
                            try {
                                detectedObjects = yolo.detect(upright)
                            } catch (e: Exception) {
                                Log.e("Yolo", "Detection error", e)
                            } finally {
                                yoloBusy.set(false)
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e("CameraPreview", "Frame processing error", e)
                } finally {
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
            camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            cameraProvider?.unbindAll()
            executor.shutdown()
            landmarker.close()
            yoloExecutor.execute { yolo.close() }
            yoloExecutor.shutdown()
            beeper.stop()
            voice.destroy()
            speaker.shutdown()
        }
    }

    // ---- Full Screen Touch Handler & UI ----
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
            .statusBarsPadding()
            .pointerInput(Unit) {
                detectTapGestures(
                    onTap = {
                        targetLabel = null
                        lockedBox = null
                        listening = false
                        hasAnnouncedFound = false // Reset here too
                        appMode = AppMode.LISTING
                    }
                )
            },
        contentAlignment = Alignment.TopCenter
    ) {
        val bmp = frame
        if (bmp != null) {
            Box(Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height)) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )
                Canvas(Modifier.fillMaxSize()) {
                    val pts = points
                    val scaleX = size.width / bmp.width
                    val scaleY = size.height / bmp.height

                    if (pts.size == 21) {
                        HAND_LINKS.forEach { (a, b) ->
                            drawLine(
                                Color.Green,
                                Offset(pts[a].x * size.width, pts[a].y * size.height),
                                Offset(pts[b].x * size.width, pts[b].y * size.height),
                                strokeWidth = 6f
                            )
                        }
                        pts.forEach { p ->
                            drawCircle(Color.Red, 10f, Offset(p.x * size.width, p.y * size.height))
                        }
                    }
                    val fingerTip = if (pts.size == 21) {
                        Offset(pts[8].x * size.width, pts[8].y * size.height)
                    } else null

                    detectedObjects.forEach { obj ->
                        val left = obj.boundingBox.left * scaleX
                        val top = obj.boundingBox.top * scaleY
                        val boxWidth = obj.boundingBox.width() * scaleX
                        val boxHeight = obj.boundingBox.height() * scaleY
                        val isTarget = obj.label == targetLabel && appMode == AppMode.TARGET_RANGE

                        drawRect(
                            color = if (isTarget) Color.Cyan else Color.Yellow,
                            topLeft = Offset(left, top),
                            size = Size(boxWidth, boxHeight),
                            style = Stroke(width = if (isTarget) 8f else 4f)
                        )

                        val textLabel = "${obj.label} ${(obj.score * 100).toInt()}%"
                        val textLayoutResult = textMeasurer.measure(
                            text = textLabel,
                            style = TextStyle(
                                color = Color.Yellow,
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold
                            )
                        )
                        val textY = if (top > textLayoutResult.size.height + 10) {
                            top - textLayoutResult.size.height - 8f
                        } else {
                            top + 8f
                        }
                        drawRect(
                            color = Color.Black.copy(alpha = 0.75f),
                            topLeft = Offset(left, textY),
                            size = Size(
                                width = textLayoutResult.size.width + 16f,
                                height = textLayoutResult.size.height + 8f
                            )
                        )
                        drawText(
                            textLayoutResult = textLayoutResult,
                            topLeft = Offset(left + 8f, textY + 4f)
                        )
                    }

                    val target = lockedBox
                    if (fingerTip != null && target != null && appMode == AppMode.TARGET_RANGE) {
                        val cx = ((target.left + target.right) / 2f) * scaleX
                        val cy = ((target.top + target.bottom) / 2f) * scaleY
                        drawLine(
                            color = Color.Magenta,
                            start = fingerTip,
                            end = Offset(cx, cy),
                            strokeWidth = 4f
                        )
                    }
                }
            }
        }

        // Diagnostic Text Display
        Text(
            "FPS: $fps   |   $lastSpoken",
            color = Color.White,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .background(Color.Black.copy(alpha = 0.6f))
                .padding(8.dp)
        )
    }
}