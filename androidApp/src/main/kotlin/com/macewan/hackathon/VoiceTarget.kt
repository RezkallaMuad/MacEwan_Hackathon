package com.macewan.hackathon

import android.content.Context
import android.content.Intent
import android.os.Bundle
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
    private val onResult: (String?) -> Unit
) {
    private val recognizer = SpeechRecognizer.createSpeechRecognizer(context)

    init {
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val heard = results
                    ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: emptyList()
                onResult(heard.firstNotNullOfOrNull { matchLabel(it) })
            }
            override fun onError(error: Int) = onResult(null)
            override fun onReadyForSpeech(p: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rms: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(p: Bundle?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
    }

    fun listen() {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) { onResult(null); return }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
        }
        recognizer.startListening(intent)
    }

    fun destroy() = recognizer.destroy()
}