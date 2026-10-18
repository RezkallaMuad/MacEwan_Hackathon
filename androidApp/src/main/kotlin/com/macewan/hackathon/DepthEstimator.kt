package com.macewan.hackathon

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * GPU delegate that saves its compiled GPU programs to disk, so only the first launch
 * pays the (slow) compile cost. The token changes if the model file changes.
 */
internal fun cachedGpuDelegate(context: Context, modelName: String, modelBytes: Long): GpuDelegate {
    val opts = GpuDelegate.Options()
    // Not codeCacheDir: Android wipes that on every reinstall/update, which would make
    // every installDebug slow again. The token below invalidates it if the model changes.
    val dir = java.io.File(context.noBackupFilesDir, "gpu_cache").apply { mkdirs() }
    opts.setSerializationParams(
        dir.absolutePath,
        "${modelName.replace('.', '_')}_${modelBytes}_tfl2161"
    )
    return GpuDelegate(opts)
}

/** A normalized depth map plus the timestamp of the camera frame it was computed from. */
class DepthFrame(val map: FloatArray, val timestampMs: Long)

class DepthEstimator(
    context: Context,
    modelName: String = "midasDepth.tflite",   // match your file exactly
    /** MiDaS small expects ImageNet mean/std. If output looks flat, try false. */
    private val imagenetNorm: Boolean = true
) {
    private val interpreter: Interpreter
    private var gpu: GpuDelegate? = null

    /** True if running on the GPU, false if it fell back to the CPU. */
    val usingGpu: Boolean get() = gpu != null

    /** How long the last inference took (ms), for the debug readout. */
    @Volatile var lastInferenceMs = 0L
        private set
    private val inW: Int
    private val inH: Int
    private val channelsFirst: Boolean
    private val input: ByteBuffer
    private val output: ByteBuffer
    private val pixels: IntArray
    val mapW: Int
    val mapH: Int

    /** Latest normalized 0..1 map (1 = closest, MiDaS inverse depth), row-major, with its frame time. */
    @Volatile var frame: DepthFrame? = null
        private set
    val map: FloatArray? get() = frame?.map

    // Normalization range smoothed across frames so the scale doesn't jump every frame.
    private var lo = Float.NaN
    private var hi = Float.NaN

    init {
        val fd = context.assets.openFd(modelName)
        val buffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        // GPU first (CPU is busy with hand tracking + frame handling); fall back to CPU.
        interpreter = try {
            val delegate = cachedGpuDelegate(context, modelName, fd.declaredLength)
            gpu = delegate
            Interpreter(buffer, Interpreter.Options().addDelegate(delegate))
        } catch (e: Throwable) {
            Log.w("Depth", "GPU delegate failed, using CPU", e)
            gpu?.close(); gpu = null
            Interpreter(buffer, Interpreter.Options().setNumThreads(2))
        }
        Log.d("Depth", "usingGpu=$usingGpu")

        val inShape = interpreter.getInputTensor(0).shape()
        channelsFirst = inShape[1] == 3
        inH = if (channelsFirst) inShape[2] else inShape[1]
        inW = if (channelsFirst) inShape[3] else inShape[2]

        val outShape = interpreter.getOutputTensor(0).shape()
        val dims = outShape.filter { it != 1 }
        mapH = dims[dims.size - 2]
        mapW = dims[dims.size - 1]
        Log.d("Depth", "in=${inW}x$inH channelsFirst=$channelsFirst out=${outShape.toList()}")

        input = ByteBuffer.allocateDirect(4 * inW * inH * 3).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * mapW * mapH).order(ByteOrder.nativeOrder())
        pixels = IntArray(inW * inH)
    }

    fun estimate(bmp: Bitmap, timestampMs: Long = SystemClock.uptimeMillis()) {
        val scaled = Bitmap.createScaledBitmap(bmp, inW, inH, true)
        scaled.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (scaled !== bmp) scaled.recycle()

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        fun v(p: Int, c: Int): Float {
            val x = ((p shr intArrayOf(16, 8, 0)[c]) and 0xFF) / 255f
            return if (imagenetNorm) (x - mean[c]) / std[c] else x
        }

        input.rewind()
        if (channelsFirst) {
            for (c in 0..2) for (p in pixels) input.putFloat(v(p, c))
        } else {
            for (p in pixels) { input.putFloat(v(p, 0)); input.putFloat(v(p, 1)); input.putFloat(v(p, 2)) }
        }
        input.rewind()
        output.rewind()
        val t0 = SystemClock.uptimeMillis()
        interpreter.run(input, output)
        lastInferenceMs = SystemClock.uptimeMillis() - t0
        output.rewind()

        val raw = FloatArray(mapW * mapH)
        output.asFloatBuffer().get(raw)
        // Robust range: 2nd/98th percentile (on a subsample) instead of raw min/max,
        // then smoothed over time. Raw min/max come from single noisy pixels and
        // shift whenever the hand (usually the closest thing) moves.
        val step = 4
        val sample = FloatArray((raw.size + step - 1) / step) { raw[it * step] }
        sample.sort()
        val pLo = sample[(sample.size * 0.02f).toInt()]
        val pHi = sample[((sample.size - 1) * 0.98f).toInt()]
        if (lo.isNaN()) { lo = pLo; hi = pHi }
        else { lo += (pLo - lo) * RANGE_SMOOTHING; hi += (pHi - hi) * RANGE_SMOOTHING }
        val range = (hi - lo).coerceAtLeast(1e-6f)
        for (i in raw.indices) raw[i] = ((raw[i] - lo) / range).coerceIn(0f, 1f)
        frame = DepthFrame(raw, timestampMs)
    }

    /** Median of a small window at normalized coords (0..1). Null if no map yet. */
    fun depthAt(nx: Float, ny: Float, f: DepthFrame? = frame): Float? {
        val m = f?.map ?: return null
        val cx = (nx * (mapW - 1)).toInt().coerceIn(0, mapW - 1)
        val cy = (ny * (mapH - 1)).toInt().coerceIn(0, mapH - 1)
        val vals = ArrayList<Float>(25)
        for (dy in -2..2) for (dx in -2..2) {
            val x = (cx + dx).coerceIn(0, mapW - 1)
            val y = (cy + dy).coerceIn(0, mapH - 1)
            vals.add(m[y * mapW + x])
        }
        vals.sort()
        return vals[vals.size / 2]
    }

    fun close() {
        interpreter.close()
        gpu?.close()
    }

    companion object {
        /** 0..1, how fast the normalization range follows each new frame. Lower = steadier. */
        const val RANGE_SMOOTHING = 0.2f
    }
}