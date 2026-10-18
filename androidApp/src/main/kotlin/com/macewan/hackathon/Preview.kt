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
import android.hardware.camera2.CameraCharacteristics
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
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
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
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
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.ExecutorService

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

private fun createLandmarker(context: Context, onResult: (List<Offset>, Long) -> Unit): HandLandmarker {
    val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
    val options = HandLandmarker.HandLandmarkerOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .setNumHands(1)
        .setResultListener { result, _ ->
            val hand = result.landmarks().firstOrNull()
            onResult(hand?.map { Offset(it.x(), it.y()) } ?: emptyList(), result.timestampMs())
        }
        .setErrorListener { e -> Log.e("HandTracker", "MediaPipe error", e) }
        .build()
    return HandLandmarker.createFromOptions(context, options)
}

/** Horizontal field of view in degrees, from the lens focal length and sensor size. */
@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
private fun horizontalFovDegrees(info: CameraInfo): Float? = try {
    val c2 = Camera2CameraInfo.from(info)
    val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull()
    val sensor = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
    if (focal == null || sensor == null || focal <= 0f) null
    else Math.toDegrees(2.0 * kotlin.math.atan(sensor.width / (2.0 * focal))).toFloat()
} catch (e: Exception) { null }

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
private fun cameraIdOf(info: CameraInfo): String =
    try { Camera2CameraInfo.from(info).cameraId } catch (e: Exception) { "?" }

/** A loaded YOLO detector and the thread it must run on. */
private class YoloSlot(val det: YoloDetector, val exec: ExecutorService)

data class DetectedItem(val label: String, val score: Float, val boundingBox: RectF)
val IGNORED_LABELS = setOf("person", "dining table")

// Wrist + finger bases: solid, interior parts of the hand, so depth there isn't mixed with background.
private val PALM_LANDMARKS = listOf(0, 1, 5, 9, 13, 17)

/** Max age gap (ms) between the depth map's frame and the hand landmarks' frame. Tune. */
private const val DEPTH_MAX_SKEW_MS = 200L
/** How strongly the depth gap counts against the 2D gap. Tune. */
private const val DEPTH_WEIGHT = 0.6f
/** Number of recent distance readings the median filter looks at. */
private const val DISTANCE_MEDIAN_WINDOW = 5

/**
 * Horizontal position in the frame (0 = left edge, 1 = right edge) -> top half of a clock:
 * left edge = 9, center = 12, right edge = 3. The user infers orientation from their grip.
 */
fun clockHour(nx: Float): Int {
    val h = 9 + Math.round(nx.coerceIn(0f, 1f) * 6f)   // 9..15
    return if (h > 12) h - 12 else h
}

/** One physical item followed across the listing scan (so two cups stay two cups). */
private class ListingTrack(val label: String, var x: Float, var y: Float) {
    val xs = mutableListOf<Float>()
    val colors = mutableListOf<String>()
    var hits = 0
    /** Color seen in at least 40% of samples, else null (too unsure to say). */
    fun color(): String? {
        val best = colors.groupingBy { it }.eachCount().maxByOrNull { it.value } ?: return null
        return if (best.value >= colors.size * 0.4f) best.key else null
    }
}

/**
 * Which detection of the target label to follow: the right color if one was asked for,
 * and otherwise the one nearest the box already locked (so it doesn't hop between two cups).
 */
private fun pickTarget(cands: List<DetectedItem>, color: String?, prev: RectF?, bmp: Bitmap): DetectedItem? {
    if (cands.isEmpty()) return null
    var pool = cands
    if (color != null) {
        val byColor = cands.filter { dominantColor(bmp, it.boundingBox) == color }
        if (byColor.isNotEmpty()) pool = byColor
        else if (prev == null) return null   // don't lock onto the wrong-colored one
    }
    return if (prev != null) {
        pool.minBy { hypot(it.boundingBox.centerX() - prev.centerX(), it.boundingBox.centerY() - prev.centerY()) }
    } else pool.maxBy { it.score }
}

private fun median(values: List<Float>): Float? {
    if (values.isEmpty()) return null
    val s = values.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2f
}

/** Per-session filter state for the hand-to-target distance. */
private class ProximityState {
    var objDepth: Float? = null
    var dDepth = 0f
    var dDepthAt = 0L
    val history = ArrayDeque<Float>()
    fun reset() { objDepth = null; dDepthAt = 0L; history.clear() }
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
    val alignment = rememberCameraAlignment()   // teammate's tilt code

    var points by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var pointsTime by remember { mutableLongStateOf(0L) }
    val prox = remember { ProximityState() }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var fps by remember { mutableIntStateOf(0) }

    // camera and torch states
    var camera by remember { mutableStateOf<Camera?>(null) }
    var isTorchOn by remember { mutableStateOf(false) }

    // tts
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }

    var detectedObjects by remember { mutableStateOf<List<DetectedItem>>(emptyList()) }
    // Models load in the background (GPU setup can take many seconds) on the same
    // thread that later runs them. Null until loaded.
    val yoloRef = remember { AtomicReference<YoloSlot?>(null) }
    val yoloGpuExecutor = remember { Executors.newSingleThreadExecutor() }
    val yoloExecutor = remember { Executors.newSingleThreadExecutor() }
    val yoloBusy = remember { AtomicBoolean(false) }
    val depthRef = remember { AtomicReference<DepthEstimator?>(null) }
    val depthExecutor = remember { Executors.newSingleThreadExecutor() }
    val depthBusy = remember { AtomicBoolean(false) }
    // Set when each model finishes loading (or fails to load), for the loading indicator.
    var yoloReady by remember { mutableStateOf(false) }
    var depthReady by remember { mutableStateOf(false) }
    // Only YOLO is needed to start; MiDaS (depth) finishes loading in the background.
    val modelsReady = yoloReady
    LaunchedEffect(Unit) {
        // 1) CPU YOLO: loads in seconds so the app is usable right away.
        yoloExecutor.execute {
            val t = SystemClock.uptimeMillis()
            try {
                val cpu = YoloDetector(context, useGpu = false)
                // If the GPU version already won the race (cache hit), don't replace it.
                if (!yoloRef.compareAndSet(null, YoloSlot(cpu, yoloExecutor))) cpu.close()
            } catch (e: Throwable) { Log.e("Yolo", "CPU load failed", e) }
            finally { yoloReady = true }
            Log.d("Yolo", "CPU loaded in ${SystemClock.uptimeMillis() - t}ms")
        }
        // 2) GPU YOLO: compiles in the background (fast if cached), then takes over.
        yoloGpuExecutor.execute {
            val t = SystemClock.uptimeMillis()
            try {
                val g = YoloDetector(context)
                if (g.usingGpu) {
                    val old = yoloRef.getAndSet(YoloSlot(g, yoloGpuExecutor))
                    // Close the CPU one on its own thread, after any detection in progress.
                    old?.let { o -> o.exec.execute { o.det.close() } }
                    yoloReady = true
                } else g.close()
            } catch (e: Throwable) { Log.e("Yolo", "GPU load failed", e) }
            Log.d("Yolo", "GPU loaded in ${SystemClock.uptimeMillis() - t}ms")
        }
        depthExecutor.execute {
            val t = SystemClock.uptimeMillis()
            try { depthRef.set(DepthEstimator(context)) } catch (e: Throwable) { Log.e("Depth", "load failed", e) }
            finally { depthReady = true }
            Log.d("Depth", "loaded in ${SystemClock.uptimeMillis() - t}ms")
        }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }
    val landmarker = remember {
        createLandmarker(context) { p, t -> pointsTime = t; points = p }
    }
    val textMeasurer = rememberTextMeasurer()

    val beeper = remember { ProximityBeeper() }
    LaunchedEffect(Unit) { beeper.start(this) }
    val chime = remember { Chime() }
    val speaker = remember { Speaker(context) }
    rememberShakeDetector(enabled = true) {
        isTorchOn = !isTorchOn
        speaker.say(if (isTorchOn) "Flashlight on" else "Flashlight off")
    }
    // --- State for Tap-Driven Flow ---
    var appMode by remember { mutableStateOf(AppMode.IDLE) }
    var isForeground by remember { mutableStateOf(true) }
    var debugText by remember { mutableStateOf("") }
    var scanning by remember { mutableStateOf(false) }
    var listening by remember { mutableStateOf(false) }
    var targetLabel by remember { mutableStateOf<String?>(null) }
    var targetColor by remember { mutableStateOf<String?>(null) }
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
        if (!yoloReady) {
            // Usually ready in a few seconds (CPU YOLO), so stay quiet unless it drags on.
            var waited = 0L
            var announced = false
            while (!yoloReady) {
                delay(250)
                waited += 250
                if (!announced && waited >= 3_000 && isForeground) {
                    speaker.say("Loading, please wait.")
                    announced = true
                }
                // Soft chime every 4 seconds after that so it doesn't seem frozen
                if (waited > 3_000 && waited % 4_000 == 0L && isForeground) chime.pending()
            }
            if (isForeground) chime.ready()
            delay(900) // let the tune finish before speaking
            speaker.say("Ready. Tap to scan.")
        } else {
            speaker.say("Tap to scan.")
        }

        while (true) {
            delay(10_000)
            if (isForeground && appMode == AppMode.IDLE && targetLabel == null) {
                speaker.say("Tap to scan.")
            }
        }
    }

    val voice = remember {
        VoiceTarget(context) { label, color ->
            listening = false
            if (label != null) {
                targetLabel = label
                targetColor = color
                appMode = AppMode.TARGET_RANGE
                hasAnnouncedFound = false // Reset the announcement tracker
                val name = listOfNotNull(color, label).joinToString(" ")
                speaker.say("Finding $name.")
                lastSpoken = "Target: $name"
            } else {
                speaker.say("Didn't catch that. Tap to try again.")
                appMode = AppMode.IDLE
                lastSpoken = "Voice failed. Tap to retry."
            }

        }
    }

    // ---- Go quiet when the screen turns off / app goes to background ----
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_STOP -> {
                    isForeground = false
                    speaker.muted = true
                    speaker.stop()
                    beeper.muted = true
                    beeper.distance = null
                    voice.cancel()
                    listening = false
                    if (appMode == AppMode.LISTING) appMode = AppMode.IDLE
                }
                Lifecycle.Event.ON_START -> {
                    isForeground = true
                    speaker.muted = false
                    beeper.muted = false
                }
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    // ---- Mode 1: Listing Mode (Triggered by Tap) ----
    LaunchedEffect(appMode) {
        if (appMode == AppMode.LISTING) {
            beeper.distance = null
            scanning = true
            listening = false
            hasAnnouncedFound = false
            lastSpoken = "Scanning..."

            // Follow each physical item across samples by label + position, so
            // multiple items of the same kind are reported separately.
            val tracks = mutableListOf<ListingTrack>()
            val samples = 20
            repeat(samples) {
                val f = frame
                if (f != null && f.width > 0 && f.height > 0) {
                    val fw = f.width.toFloat(); val fh = f.height.toFloat()
                    val used = HashSet<ListingTrack>()
                    for (d in detectedObjects.sortedByDescending { it.score }) {
                        val nx = d.boundingBox.centerX() / fw
                        val ny = d.boundingBox.centerY() / fh
                        val t = tracks
                            .filter { it.label == d.label && it !in used &&
                                    kotlin.math.abs(it.x - nx) < 0.1f && kotlin.math.abs(it.y - ny) < 0.15f }
                            .minByOrNull { kotlin.math.abs(it.x - nx) + kotlin.math.abs(it.y - ny) }
                            ?: ListingTrack(d.label, nx, ny).also { tracks.add(it) }
                        used.add(t)
                        t.hits++
                        t.xs.add(nx)
                        t.x += (nx - t.x) * 0.3f
                        t.y += (ny - t.y) * 0.3f
                        dominantColor(f, d.boundingBox)?.let { t.colors.add(it) }
                    }
                }
                delay(100)
            }
            scanning = false

            // Items seen consistently, ordered left (9 o'clock) to right (3 o'clock)
            val parts = tracks.filter { it.hits >= samples * 0.3 }
                .sortedBy { median(it.xs) ?: 0.5f }
                .map { t ->
                    val name = listOfNotNull(t.color(), t.label).joinToString(" ")
                    "$name, ${clockHour(median(t.xs) ?: 0.5f)} o'clock"
                }

            if (parts.isNotEmpty()) {
                val msg = parts.joinToString(". ") + ". Which one?"
                speaker.say(msg)
                lastSpoken = parts.joinToString(". ")

                delay((msg.length * 80L) + 1000L)

                if (micOk && appMode == AppMode.LISTING) {
                    listening = true
                    lastSpoken = "Listening for your choice..."
                    voice.listen()
                }
            } else {
                val msg = "Nothing found. Tap to try again."
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
            prox.reset()
            return@LaunchedEffect
        }

        val pts = points
        val now = SystemClock.uptimeMillis()
        val w = bmp.width.toFloat(); val h = bmp.height.toFloat()

        // Padded hand bounding box in pixels, used to tell when the hand covers the target.
        val handBox: RectF? = if (pts.size == 21) {
            val pad = 0.04f * w
            RectF(
                pts.minOf { it.x } * w - pad, pts.minOf { it.y } * h - pad,
                pts.maxOf { it.x } * w + pad, pts.maxOf { it.y } * h + pad
            )
        } else null
        val handOverLocked = lockedBox?.let { lb -> handBox != null && RectF.intersects(lb, handBox) } ?: false

        val match = pickTarget(detectedObjects.filter { it.label == targetLabel }, targetColor, lockedBox, bmp)
        if (match != null) {
            // While the hand covers the target, YOLO's box shrinks and shifts with the
            // occlusion. Keep the last clean box instead of following it.
            if (lockedBox == null || !handOverLocked) lockedBox = match.boundingBox
            lockedAt = now

            // NEW: If we just found it and haven't announced it yet, say it!
            if (!hasAnnouncedFound) {
                val name = listOfNotNull(targetColor, targetLabel).joinToString(" ")
                speaker.say("$name, ${clockHour(match.boundingBox.centerX() / bmp.width)} o'clock.")
                hasAnnouncedFound = true
            }

        } else if (handOverLocked) {
            // Hand is hiding the object from YOLO; keep the lock alive.
            lockedAt = now
        } else if (now - lockedAt > 3000) {
            lockedBox = null
            prox.reset()
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
                val d = hypot(p.x * bmp.width - cx, p.y * bmp.height - cy)
                if (d < best) best = d
            }
            val d2d = (best / maxDist).coerceIn(0f, 1f)

            // Only trust the depth map if it came from (nearly) the same camera frame
            // as the hand landmarks; otherwise the hand has moved and we'd sample background.
            val dep = depthRef.get()
            val df = dep?.frame
            if (dep != null && df != null && kotlin.math.abs(df.timestampMs - pointsTime) <= DEPTH_MAX_SKEW_MS) {
                // Hand depth: median over palm landmarks (not a single fingertip on the hand's edge).
                val handDepth = median(PALM_LANDMARKS.mapNotNull { dep.depthAt(pts[it].x, pts[it].y, df) })

                // Object depth: 3x3 grid inside the box, skipping points the hand covers.
                // If the hand covers all of them, keep the last clean reading.
                val objSamples = ArrayList<Float>(9)
                for (fy in floatArrayOf(0.2f, 0.5f, 0.8f)) for (fx in floatArrayOf(0.2f, 0.5f, 0.8f)) {
                    val px = box.left + fx * box.width()
                    val py = box.top + fy * box.height()
                    if (handBox != null && handBox.contains(px, py)) continue
                    dep.depthAt(px / w, py / h, df)?.let { objSamples.add(it) }
                }
                median(objSamples)?.let { prox.objDepth = it }

                val objDepth = prox.objDepth
                if (handDepth != null && objDepth != null) {
                    val gap = kotlin.math.abs(handDepth - objDepth)
                    // Smooth the depth gap itself; it's the noisiest input.
                    prox.dDepth = if (prox.dDepthAt == 0L) gap else prox.dDepth + (gap - prox.dDepth) * 0.3f
                    prox.dDepthAt = now
                }
            }

            // Keep using the last depth gap for a while when this update has no fresh depth,
            // instead of flipping between "with depth" and "2D only" (which jumps the value).
            val useDepth = prox.dDepthAt != 0L && now - prox.dDepthAt <= 2000
            result = if (useDepth) {
                hypot(d2d, DEPTH_WEIGHT * prox.dDepth).coerceIn(0f, 1f)
            } else d2d

            debugText = "2D %.2f  depth %s  skew %s  -> %.2f\nYOLO %s  MiDaS %s %dms".format(
                d2d,
                if (useDepth) "%.2f".format(prox.dDepth) else "off",
                df?.let { "${kotlin.math.abs(it.timestampMs - pointsTime)}ms" } ?: "-",
                result,
                if (yoloRef.get()?.det?.usingGpu == true) "GPU" else "CPU",
                when { dep == null -> "loading"; dep.usingGpu -> "GPU"; else -> "CPU" },
                dep?.lastInferenceMs ?: 0L
            )
        }

        // Median filter: drop single-frame spikes before the beeper's smoothing.
        if (result == null) {
            prox.history.clear()
            beeper.distance = null
        } else {
            prox.history.addLast(result)
            while (prox.history.size > DISTANCE_MEDIAN_WINDOW) prox.history.removeFirst()
            beeper.distance = median(prox.history.toList())
        }
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
                    val frameTs = currentTimestamp

                    landmarker.detectAsync(mpImage, currentTimestamp)
                    val slot = yoloRef.get()
                    if (slot != null && yoloBusy.compareAndSet(false, true)) {
                        slot.exec.execute {
                            try {
                                detectedObjects = slot.det.detect(upright)
                            } catch (e: Exception) {
                                Log.e("Yolo", "Detection error", e)
                            } finally {
                                yoloBusy.set(false)
                            }
                        }
                    }
                    if (depthBusy.compareAndSet(false, true)) {
                        depthExecutor.execute {
                            try { depthRef.get()?.estimate(upright, frameTs) }
                            catch (e: Exception) { Log.e("Depth", "error", e) }
                            finally { depthBusy.set(false) }
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

            // 1. Build a custom selector that hunts for the widest lens available
            // 1. Build a custom selector that hunts for the widest lens available
            val wideCameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .addCameraFilter { cameraInfos ->
                    Log.d("CameraLenses", "Total back cameras found by CameraX: ${cameraInfos.size}")

                    cameraInfos.forEach { info ->
                        val minZoom = info.zoomState.value?.minZoomRatio ?: "Unknown"
                        val maxZoom = info.zoomState.value?.maxZoomRatio ?: "Unknown"
                        Log.d("CameraLenses", "Camera ${cameraIdOf(info)} -> FOV: ${horizontalFovDegrees(info)} deg, minZoom: $minZoom, maxZoom: $maxZoom")
                    }

                    // Pick the lens with the widest field of view (the ultra-wide, if the
                    // phone exposes it as its own camera). Zoom ratio can't tell these apart,
                    // since every separate camera reports 1.0x as its minimum.
                    val widestCamera = cameraInfos.maxByOrNull { info ->
                        val fov = horizontalFovDegrees(info) ?: 0f
                        // Tie-break: a lower min zoom (e.g. 0.5x on a logical multi-camera) is wider.
                        fov / (info.zoomState.value?.minZoomRatio ?: 1f)
                    }

                    Log.d("CameraLenses", "Selected camera ${widestCamera?.let { cameraIdOf(it) }} (FOV ${widestCamera?.let { horizontalFovDegrees(it) }} deg)")

                    listOfNotNull(widestCamera ?: cameraInfos.firstOrNull())
                }
                .build()

            // 2. Bind using our custom wide selector instead of the default
            val boundCamera = provider.bindToLifecycle(lifecycleOwner, wideCameraSelector, analysis)
            camera = boundCamera

            // 3. Use the widest zoom the camera supports (e.g. 0.5x on phones with an
            //    ultra-wide). Re-applied whenever zoom drifts, e.g. after screen off/on.
            boundCamera.cameraInfo.zoomState.observe(lifecycleOwner) { z ->
                if (z.zoomRatio > z.minZoomRatio + 0.01f) {
                    Log.d("CameraLenses", "Zoom range ${z.minZoomRatio}x-${z.maxZoomRatio}x, setting ${z.minZoomRatio}x")
                    boundCamera.cameraControl.setZoomRatio(z.minZoomRatio)
                }
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            cameraProvider?.unbindAll()
            executor.shutdown()
            landmarker.close()
            yoloRef.get()?.let { s -> s.exec.execute { s.det.close() } }
            yoloExecutor.shutdown()
            yoloGpuExecutor.shutdown()
            beeper.stop()
            chime.release()
            voice.destroy()
            speaker.shutdown()
            depthExecutor.execute { depthRef.get()?.close() }
            depthExecutor.shutdown()
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
                    onTap = onTap@{
                        if (!(yoloReady)) {
                            chime.pending()
                            return@onTap
                        }
                        targetLabel = null
                        targetColor = null
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
                    val fingerTip = if (pts.size == 21 && lockedBox != null) {
                        val t = lockedBox!!
                        val cx = ((t.left + t.right) / 2f) * scaleX
                        val cy = ((t.top + t.bottom) / 2f) * scaleY
                        pts.map { Offset(it.x * size.width, it.y * size.height) }
                            .minByOrNull { hypot(it.x - cx, it.y - cy) }
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

        // Loading indicator
        if (!modelsReady) {
            Column(
                Modifier
                    .align(Alignment.Center)
                    .background(Color.Black.copy(alpha = 0.75f), RoundedCornerShape(16.dp))
                    .padding(28.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                CircularProgressIndicator(color = Color.White)
                Spacer(Modifier.height(16.dp))
                Text("Loading...", color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Preparing object detection",
                    color = Color.White.copy(alpha = 0.8f),
                    fontSize = 16.sp
                )
            }
        }

        // Diagnostic Text Display
        Text(
            "FPS: $fps   |   $lastSpoken" + if (debugText.isNotEmpty()) "\n$debugText" else "",
            color = Color.White,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .background(Color.Black.copy(alpha = 0.6f))
                .padding(8.dp)
        )
    }
}