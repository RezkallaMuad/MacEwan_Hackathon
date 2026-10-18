package com.macewan.hackathon

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue

// The 80 labels the detector knows
val COCO_LABELS = setOf(
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

// Things people say that the model calls something else
private val SYNONYMS = mapOf(
    "mug" to "cup", "phone" to "cell phone", "table" to "dining table",
    "sofa" to "couch", "television" to "tv", "plant" to "potted plant",
    "teddy" to "teddy bear", "ball" to "sports ball", "doughnut" to "donut",
    "fridge" to "refrigerator", "glass" to "wine glass", "controller" to "remote"
)

/** Turns "find the bottles" into "bottle", or null if no known object was said. */
fun matchLabel(spoken: String): String? {
    val text = spoken.lowercase()
    val keys = COCO_LABELS.map { it to it } + SYNONYMS.map { it.key to it.value }
    // longest match wins, so "hot dog" beats "dog" and "cell phone" beats "phone"
    return keys.filter { text.contains(it.first) }.maxByOrNull { it.first.length }?.second
}

class VoiceTarget(
    private val context: Context,
    /** (label, color) — color is null if none was said. Both null if nothing matched. */
    private val onResult: (String?, String?) -> Unit
) {
    private val recognizer = SpeechRecognizer.createSpeechRecognizer(context)
    private val handler = Handler(Looper.getMainLooper())
    private var active = false
    private var deadline = 0L

    private val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
        // Ask the recognizer to wait longer for silence. Some recognizers ignore these,
        // which is why we also restart listening below until the window runs out.
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2500L)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 2500L)
    }

    /** Restart listening if there's still time left in the window. Returns false if time is up. */
    private fun retry(): Boolean {
        if (!active || SystemClock.uptimeMillis() >= deadline) return false
        handler.postDelayed({ if (active) recognizer.startListening(intent) }, 200)
        return true
    }

    private fun finish(label: String?, color: String? = null) {
        if (!active) return
        active = false
        onResult(label, color)
    }

    init {
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val heard = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: emptyList()
                val hit = heard.firstNotNullOfOrNull { h -> matchLabel(h)?.let { it to matchColor(h, it) } }
                // Heard something that isn't an object? Keep listening while there's time.
                if (hit != null) finish(hit.first, hit.second)
                else if (!retry()) finish(null)
            }
            override fun onError(error: Int) {
                if (!active) return
                val retryable = error == SpeechRecognizer.ERROR_NO_MATCH ||
                        error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT ||
                        error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY ||
                        error == SpeechRecognizer.ERROR_CLIENT
                if (!retryable || !retry()) finish(null)
            }
            override fun onReadyForSpeech(p: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rms: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(p: Bundle?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
    }

    /** Listen for an object name, retrying through silence for up to [windowMs]. */
    fun listen(windowMs: Long = LISTEN_WINDOW_MS) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) { onResult(null, null); return }
        handler.removeCallbacksAndMessages(null)
        active = true
        deadline = SystemClock.uptimeMillis() + windowMs
        recognizer.startListening(intent)
    }

    /** Stop listening without reporting a result. */
    fun cancel() {
        active = false
        handler.removeCallbacksAndMessages(null)
        recognizer.cancel()
    }

    fun destroy() {
        cancel()
        recognizer.destroy()
    }

    companion object {
        /** Total time to keep listening for an object name. */
        const val LISTEN_WINDOW_MS = 8000L
    }
}