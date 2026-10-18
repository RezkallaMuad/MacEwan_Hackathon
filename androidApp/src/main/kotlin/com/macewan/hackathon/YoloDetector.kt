package com.macewan.hackathon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class YoloDetector(
    context: Context,
    modelName: String = "yolo11s_float32.tflite",   // must match your file in assets/ exactly
    private val scoreThreshold: Float = 0.45f,
    private val iouThreshold: Float = 0.5f,
    private val maxResults: Int = 10,
    /** Boxes as 0..1. If boxes look tiny or stuck top-left, set to false. */
    private val normalizedOutput: Boolean = true,
    /** Try the GPU (slow first load, fast inference). False = CPU only, loads in seconds. */
    useGpu: Boolean = true
) {
    private val interpreter: Interpreter
    private var gpu: GpuDelegate? = null

    /** True if running on the GPU. */
    val usingGpu: Boolean get() = gpu != null
    private val inW: Int
    private val inH: Int
    private val numAnchors: Int
    private val numClasses: Int
    private val input: ByteBuffer
    private val output: Array<Array<FloatArray>>
    private val pixels: IntArray

    private val channelsFirst: Boolean
    init {
        val fd = context.assets.openFd(modelName)
        val buffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)

        // GPU first; if the delegate or the model fails on it, fall back to CPU.
        var delegate: GpuDelegate? = null
        interpreter = if (!useGpu) {
            Interpreter(buffer, Interpreter.Options().setNumThreads(4))
        } else try {
            delegate = cachedGpuDelegate(context, modelName, fd.declaredLength)
            Interpreter(buffer, Interpreter.Options().addDelegate(delegate))
        } catch (e: Throwable) {
            Log.w("Yolo", "GPU delegate failed, using CPU", e)
            delegate?.close()
            delegate = null
            Interpreter(buffer, Interpreter.Options().setNumThreads(4))
        }
        gpu = delegate
        Log.d("Yolo", "usingGpu=${gpu != null}")

        val inShape = interpreter.getInputTensor(0).shape()
        channelsFirst = inShape[1] == 3                    // [1,3,H,W] vs [1,H,W,3]
        inH = if (channelsFirst) inShape[2] else inShape[1]
        inW = if (channelsFirst) inShape[3] else inShape[2]
        val outShape = interpreter.getOutputTensor(0).shape()    // [1, 4+classes, anchors]
        numClasses = outShape[1] - 4
        numAnchors = outShape[2]
        Log.d("Yolo", "in=${inW}x${inH} out=${outShape.toList()} classes=$numClasses anchors=$numAnchors")

        input = ByteBuffer.allocateDirect(4 * inW * inH * 3).order(ByteOrder.nativeOrder())
        output = Array(1) { Array(outShape[1]) { FloatArray(numAnchors) } }
        pixels = IntArray(inW * inH)
    }

    /** [bmp] is the upright frame. Returned boxes are in that bitmap's pixel coordinates. */
    fun detect(bmp: Bitmap): List<DetectedItem> {
        val scaled = Bitmap.createScaledBitmap(bmp, inW, inH, true)
        scaled.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        if (scaled !== bmp) scaled.recycle()

        input.rewind()
        if (channelsFirst) {
            // planar: all R values, then all G, then all B
            for (shift in intArrayOf(16, 8, 0)) {
                for (p in pixels) input.putFloat(((p shr shift) and 0xFF) / 255f)
            }
        } else {
            for (p in pixels) {
                input.putFloat(((p shr 16) and 0xFF) / 255f)
                input.putFloat(((p shr 8) and 0xFF) / 255f)
                input.putFloat((p and 0xFF) / 255f)
            }
        }
        input.rewind()

        interpreter.run(input, output)

        val out = output[0]
        val sx = if (normalizedOutput) bmp.width.toFloat() else bmp.width / inW.toFloat()
        val sy = if (normalizedOutput) bmp.height.toFloat() else bmp.height / inH.toFloat()

        val candidates = ArrayList<DetectedItem>()
        for (i in 0 until numAnchors) {
            var best = 0f
            var bestC = -1
            for (c in 0 until numClasses) {
                val s = out[4 + c][i]
                if (s > best) { best = s; bestC = c }
            }
            if (best < scoreThreshold || bestC < 0) continue
            val label = COCO.getOrNull(bestC) ?: continue
            if (label in IGNORED_LABELS) continue

            val cx = out[0][i]; val cy = out[1][i]
            val w = out[2][i]; val h = out[3][i]
            candidates.add(
                DetectedItem(
                    label = label,
                    score = best,
                    boundingBox = RectF(
                        ((cx - w / 2f) * sx).coerceIn(0f, bmp.width.toFloat()),
                        ((cy - h / 2f) * sy).coerceIn(0f, bmp.height.toFloat()),
                        ((cx + w / 2f) * sx).coerceIn(0f, bmp.width.toFloat()),
                        ((cy + h / 2f) * sy).coerceIn(0f, bmp.height.toFloat())
                    )
                )
            )
        }
        return nms(candidates)
    }

    private fun nms(dets: List<DetectedItem>): List<DetectedItem> {
        val keep = ArrayList<DetectedItem>()
        for (d in dets.sortedByDescending { it.score }) {
            if (keep.none { it.label == d.label && iou(it.boundingBox, d.boundingBox) > iouThreshold }) {
                keep.add(d)
                if (keep.size >= maxResults) break
            }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iw = minOf(a.right, b.right) - maxOf(a.left, b.left)
        val ih = minOf(a.bottom, b.bottom) - maxOf(a.top, b.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    fun close() {
        interpreter.close()
        gpu?.close()
    }

    companion object {
        val COCO = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}