package com.macewan.hackathon

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.hardware.camera2.CameraCharacteristics
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.hypot
import kotlinx.coroutines.delay

enum class AppMode {
    IDLE, LISTING, TARGET_RANGE
}

//root entry composable that checks for runtime camera permissions before rendering the camera feed
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
            Text("Camera permission needed", color = Color.White)
        }
        LaunchedEffect(Unit) { launcher.launch(Manifest.permission.CAMERA) }
    }
}

//factory for mediapipe hand landmarker, configured in LIVE_STREAM mode (runs async on background thread so camera doesnt stutter)
private fun createLandmarker(context: Context, onResult: (List<Offset>, Long) -> Unit): HandLandmarker {
    val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
    val options = HandLandmarker.HandLandmarkerOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .setNumHands(1)
        .setResultListener { result, _ ->
            val hand = result.landmarks().firstOrNull()
            //map the 3d media pipe landmarker to the 2d normalized offsets on screen
            onResult(hand?.map { Offset(it.x(), it.y()) } ?: emptyList(), result.timestampMs())
        }
        .setErrorListener { e -> Log.e("HandTracker", "MediaPipe error", e) }
        .build()
    return HandLandmarker.createFromOptions(context, options)
}

//find the Horz. fov in degrees using camera2 | used automatically to pick the widest lens
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

//thread safe container pairing a loaded YOLO network with its dedicated worker thread
private class YoloSlot(val det: YoloDetector, val exec: ExecutorService)

//unified representation of oject YOLO
data class DetectedItem(val label: String, val score: Float, val boundingBox: RectF)
val IGNORED_LABELS = setOf("person", "dining table")

private val PALM_LANDMARKS = listOf(0, 1, 5, 9, 13, 17)
private const val DEPTH_MAX_SKEW_MS = 250L
private const val DEPTH_WEIGHT = 0.6f
private const val DISTANCE_MEDIAN_WINDOW = 3

fun clockHour(nx: Float): Int {
    val h = 9 + Math.round(nx.coerceIn(0f, 1f) * 6f)
    return if (h > 12) h - 12 else h
}

//tracks an obj across multple frames during the scan and prevents duplicates
private class ListingTrack(val label: String, var x: Float, var y: Float) {
    val xs = mutableListOf<Float>()
    val colors = mutableListOf<String>()
    var hits = 0
    fun color(): String? { //returns the dominant color >= 40% of frames
        val best = colors.groupingBy { it }.eachCount().maxByOrNull { it.value } ?: return null
        return if (best.value >= colors.size * 0.4f) best.key else null
    }
}

//selects which box is the active target, matches requested color and locks onto candidate nearest to prev locked box to avoid hopping between 2
private fun pickTarget(cands: List<DetectedItem>, color: String?, prev: RectF?, bmp: Bitmap): DetectedItem? {
    if (cands.isEmpty()) return null
    var pool = cands
    if (color != null) {
        val byColor = cands.filter { dominantColor(bmp, it.boundingBox) == color }
        if (byColor.isNotEmpty()) pool = byColor
        else if (prev == null) return null
    }
    return if (prev != null) {
        pool.minByOrNull { hypot(it.boundingBox.centerX() - prev.centerX(), it.boundingBox.centerY() - prev.centerY()) }
    } else pool.maxByOrNull { it.score }
}

//utility for median of a float collection
private fun median(values: List<Float>): Float? {
    if (values.isEmpty()) return null
    val s = values.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2f
}

//session state tracking hand to object depth differential
private class ProximityState {
    var objDepth: Float? = null
    var dDepth = 0f
    var dDepthAt = 0L
    val history = ArrayDeque<Float>()
    fun reset() { objDepth = null; dDepthAt = 0L; history.clear() }
}

//points on hand
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

    val alignment = rememberCameraAlignment()//phone tilt sensor

    //hand tracking state
    var points by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var pointsTime by remember { mutableLongStateOf(0L) }
    val prox = remember { ProximityState() }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var fps by remember { mutableIntStateOf(0) }

    //flashlight state
    var camera by remember { mutableStateOf<Camera?>(null) }
    var isTorchOn by remember { mutableStateOf(false) }

    var detectedObjects by remember { mutableStateOf<List<DetectedItem>>(emptyList()) } //detected objects state

    //YOLO stuff (model threading)
    val yoloRef = remember { AtomicReference<YoloSlot?>(null) }
    val yoloGpuExecutor = remember { Executors.newSingleThreadExecutor() }
    val yoloExecutor = remember { Executors.newSingleThreadExecutor() }
    val yoloBusy = remember { AtomicBoolean(false) }
    val depthRef = remember { AtomicReference<DepthEstimator?>(null) }
    val depthExecutor = remember { Executors.newSingleThreadExecutor() }
    val depthBusy = remember { AtomicBoolean(false) }

    var yoloReady by remember { mutableStateOf(false) }
    var depthReady by remember { mutableStateOf(false) }
    val modelsReady = yoloReady

    //background model launcher
    LaunchedEffect(Unit) {
        yoloExecutor.execute {
            val t = SystemClock.uptimeMillis()
            try {
                val cpu = YoloDetector(context, useGpu = false)
                if (!yoloRef.compareAndSet(null, YoloSlot(cpu, yoloExecutor))) cpu.close()
            } catch (e: Throwable) { Log.e("Yolo", "CPU load failed", e) }
            finally { yoloReady = true }
            Log.d("Yolo", "CPU loaded in ${SystemClock.uptimeMillis() - t}ms")
        }

        //yolo
        yoloGpuExecutor.execute {
            val t = SystemClock.uptimeMillis()
            try {
                val g = YoloDetector(context)
                if (g.usingGpu) {
                    val old = yoloRef.getAndSet(YoloSlot(g, yoloGpuExecutor))
                    old?.let { o -> o.exec.execute { o.det.close() } }
                    yoloReady = true
                } else g.close()
            } catch (e: Throwable) { Log.e("Yolo", "GPU load failed", e) }
            Log.d("Yolo", "GPU loaded in ${SystemClock.uptimeMillis() - t}ms")
        }

        // midas depth sensor
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

    //audio
    val beeper = remember { ProximityBeeper() }
    LaunchedEffect(Unit) { beeper.start(this) }
    val chime = remember { Chime() }
    val speaker = remember { Speaker(context) }

    //shake gesture
    rememberShakeDetector(enabled = true) {
        isTorchOn = !isTorchOn
        speaker.say(if (isTorchOn) "Flashlight on" else "Flashlight off")
    }

    // App state
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
    var hasAnnouncedFound by remember { mutableStateOf(false) }

    //mic perms
    var micOk by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val micLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { micOk = it }

    //sync torch to statechanges
    LaunchedEffect(camera, isTorchOn) {
        val cam = camera ?: return@LaunchedEffect
        if (cam.cameraInfo.hasFlashUnit()) {
            cam.cameraControl.enableTorch(isTorchOn)
        }
    }

    //initial audio boot up
    LaunchedEffect(Unit) {
        if (!micOk) micLauncher.launch(Manifest.permission.RECORD_AUDIO)

        delay(800)
        if (!yoloReady) {
            var waited = 0L
            var announced = false
            while (!yoloReady) {
                delay(250)
                waited += 250
                if (!announced && waited >= 3_000 && isForeground) {
                    speaker.say("Loading, please wait.")
                    announced = true
                }
                if (waited > 3_000 && waited % 4_000 == 0L && isForeground) chime.pending()
            }
            if (isForeground) chime.ready()
            delay(900)
            speaker.say("Ready. Tap to scan.")
        } else {
            speaker.say("Tap to scan.")
        }

    }
//    // voice looping when idle
//    LaunchedEffect(appMode) {
//        // only run when idle and ready
//        if (appMode == AppMode.IDLE) {
//            while (true) {
//                delay(12_000L) // wait 12 secs
//                if (isForeground && appMode == AppMode.IDLE && targetLabel == null && yoloReady) {
//                    speaker.say("Tap to scan.")
//                }
//            }
//        }
//    }

    //voice recognition callback - fires when user selects target
    val voice = remember {
        VoiceTarget(context) { label, color ->
            listening = false
            if (label != null) {
                targetLabel = label
                targetColor = color
                appMode = AppMode.TARGET_RANGE
                hasAnnouncedFound = false
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

    //mutes audio and stops interface when in background
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

    // ---- Mode 1: Listing Mode ----
    LaunchedEffect(appMode) {
        if (appMode == AppMode.LISTING) {
            beeper.distance = null
            scanning = true
            listening = false
            hasAnnouncedFound = false
            lastSpoken = "Scanning..."

            val tracks = mutableListOf<ListingTrack>()
            val samples = 15 // reduced from 20 to speed up responsiveness (1.5s total)
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
                                    kotlin.math.abs(it.x - nx) < 0.12f && kotlin.math.abs(it.y - ny) < 0.18f }
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

            //order items from left to right 9 to 3
            val parts = tracks.filter { it.hits >= samples * 0.25 }
                .sortedBy { median(it.xs) ?: 0.5f }
                .map { t ->
                    val name = listOfNotNull(t.color(), t.label).joinToString(" ")
                    "$name, ${clockHour(median(t.xs) ?: 0.5f)} o'clock"
                }

            if (parts.isNotEmpty()) {
                val msg = parts.joinToString(". ") + ". Select item."
                speaker.say(msg)
                lastSpoken = parts.joinToString(". ")

                delay((msg.length * 75L) + 800L)

                //prompt user choice
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

        //calculate hand bounding box
        val handBox: RectF? = if (pts.size == 21) {
            val pad = 0.04f * w
            RectF(
                pts.minOf { it.x } * w - pad, pts.minOf { it.y } * h - pad,
                pts.maxOf { it.x } * w + pad, pts.maxOf { it.y } * h + pad
            )
        } else null
        //detect occlusion - check if the hand covers the object
        val handOverLocked = lockedBox?.let { lb -> handBox != null && RectF.intersects(lb, handBox) } ?: false

        //match candidate boxes
        val match = pickTarget(detectedObjects.filter { it.label == targetLabel }, targetColor, lockedBox, bmp)
        if (match != null) {
            if (lockedBox == null || !handOverLocked) lockedBox = match.boundingBox
            lockedAt = now //keep lock even if hand covers object from YOLO sight

            if (!hasAnnouncedFound) {
                val name = listOfNotNull(targetColor, targetLabel).joinToString(" ")
                speaker.say("$name, ${clockHour(match.boundingBox.centerX() / bmp.width)} o'clock.")
                hasAnnouncedFound = true
            }
        } else if (handOverLocked) {
            lockedAt = now
        } else if (now - lockedAt > 3000) {
            lockedBox = null
            prox.reset()
            hasAnnouncedFound = false
        }

        val box = lockedBox
        var result: Float? = null

        //compute 2d euclidean distance and 3d depth
        if (pts.size == 21 && box != null) {
            val cx = (box.left + box.right) / 2f
            val cy = (box.top + box.bottom) / 2f
            val maxDist = hypot(bmp.width.toFloat(), bmp.height.toFloat()) * 0.5f

            //find closest hand point to object center
            var best = Float.MAX_VALUE
            for (p in pts) {
                val d = hypot(p.x * bmp.width - cx, p.y * bmp.height - cy)
                if (d < best) best = d
            }
            val d2d = (best / maxDist).coerceIn(0f, 1f)

            val dep = depthRef.get()
            val df = dep?.frame
            if (dep != null && df != null && kotlin.math.abs(df.timestampMs - pointsTime) <= DEPTH_MAX_SKEW_MS) {
                val handDepth = median(PALM_LANDMARKS.mapNotNull { dep.depthAt(pts[it].x, pts[it].y, df) })

                //smaple grid box
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
                    prox.dDepth = if (prox.dDepthAt == 0L) gap else prox.dDepth + (gap - prox.dDepth) * 0.3f
                    prox.dDepthAt = now
                }
            }

            //combine 2d distance and 3d depth into a distance metric
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

        //pass distance to proximity beeper through the median filter
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
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) //drop old frames
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frames = 0
            var windowStart = SystemClock.uptimeMillis()

            //analyzer callback for each live frame
            analysis.setAnalyzer(executor) { proxy ->
                try {
                    val raw = proxy.toBitmap()
                    //rotate bitmap to match portrait orientation
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

                    //run hand landmarking
                    landmarker.detectAsync(mpImage, currentTimestamp)

                    // Execute YOLO and filter out unwanted labels (person, table)
                    val slot = yoloRef.get()
                    if (slot != null && yoloBusy.compareAndSet(false, true)) {
                        slot.exec.execute {
                            try {
                                val rawDetections = slot.det.detect(upright)
                                // filter out persons and tables
                                detectedObjects = rawDetections.filter {
                                    it.label.lowercase() !in IGNORED_LABELS
                                }
                            } catch (e: Exception) {
                                Log.e("Yolo", "Detection error", e)
                            } finally {
                                yoloBusy.set(false)
                            }
                        }
                    }

                    //run midas depth estimate (async on worker thread)
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
                    proxy.close() //stop camera streaming
                }

                //fps counter
                frames++
                val now = SystemClock.uptimeMillis()
                if (now - windowStart >= 1000) {
                    fps = frames
                    frames = 0
                    windowStart = now
                }
            }
            provider.unbindAll()

            //dynamic lens selection
            val wideCameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .addCameraFilter { cameraInfos ->
                    val widestCamera = cameraInfos.maxByOrNull { info ->
                        val fov = horizontalFovDegrees(info) ?: 0f
                        fov / (info.zoomState.value?.minZoomRatio ?: 1f)
                    }
                    listOfNotNull(widestCamera ?: cameraInfos.firstOrNull())
                }
                .build()


            val boundCamera = provider.bindToLifecycle(lifecycleOwner, wideCameraSelector, analysis)
            camera = boundCamera
            //zoom out to min
            boundCamera.cameraInfo.zoomState.observe(lifecycleOwner) { z ->
                if (z.zoomRatio > z.minZoomRatio + 0.01f) {
                    boundCamera.cameraControl.setZoomRatio(z.minZoomRatio)
                }
            }
        }, ContextCompat.getMainExecutor(context))

        //clean it up on disposal
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
            //global screen tapping handler
            .pointerInput(modelsReady, appMode) {
                detectTapGestures(
                    onTap = {
                        if (!modelsReady) {
                            chime.pending()
                            return@detectTapGestures
                        }

                        if (appMode == AppMode.TARGET_RANGE) {
                            // If in target focus mode, tap exits focus mode back to IDLE
                            appMode = AppMode.IDLE
                            targetLabel = null
                            targetColor = null
                            lockedBox = null
                            beeper.distance = null
                            prox.reset()
                            hasAnnouncedFound = false
                            speaker.say("Exited target mode. Tap to scan.")
                            lastSpoken = "Tap to scan."
                        } else {
                            // In IDLE, tap starts listing scan
                            targetLabel = null
                            targetColor = null
                            lockedBox = null
                            listening = false
                            hasAnnouncedFound = false
                            appMode = AppMode.LISTING
                        }
                    },
                    onLongPress = {
                        // reset any active target so beeping stops while speaking
                        appMode = AppMode.IDLE
                        targetLabel = null
                        targetColor = null
                        lockedBox = null
                        beeper.distance = null
                        prox.reset()
                        hasAnnouncedFound = false

                        // audible info
                        chime.ready()

                        // read out the instructions
                        val instructions = "Instructions: Tilt the phone to align the screen until vibration." +
                                "Tap once to scan the table. " +
                                "Speak an item name to lock on. " +
                                "Reach forward; the beeper speeds up as you get closer. " +
                                "Tap once to cancel. " +
                                "Shake the phone to toggle the flashlight. " +
                                "Hold the screen anytime to hear this again."

                        speaker.say(instructions)
                        lastSpoken = "Reading instructions..."
                    }
                )
            },
        contentAlignment = Alignment.TopCenter
    ) {
        val bmp = frame
        if (bmp != null) {
            Box(Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height)) {
                //live cam feed
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )
                //visual overlays
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

                    //reaching fingertip
                    val fingerTip = if (pts.size == 21 && lockedBox != null) {
                        val t = lockedBox!!
                        val cx = ((t.left + t.right) / 2f) * scaleX
                        val cy = ((t.top + t.bottom) / 2f) * scaleY
                        pts.map { Offset(it.x * size.width, it.y * size.height) }
                            .minByOrNull { hypot(it.x - cx, it.y - cy) }
                    } else null

                    //draw bounding boxes for objs
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

                        //box labels
                        val textLabel = "${obj.label} ${(obj.score * 100).toInt()}%"
                        val textLayoutResult = textMeasurer.measure(
                            text = textLabel,
                            style = TextStyle(
                                color = if (isTarget) Color.Cyan else Color.Yellow,
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

                    //distance between closest and finger
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

        // live diagnostic display
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