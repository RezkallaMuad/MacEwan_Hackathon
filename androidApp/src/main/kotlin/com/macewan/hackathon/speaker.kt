package com.macewan.hackathon

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class Speaker(context: Context) {
    private var tts: TextToSpeech? = null
    @Volatile private var ready = false

    // Hold messages requested before TTS is fully initialized
    private val pendingMessages = mutableListOf<String>()

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Set language, falling back to default if US isn't available
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("Speaker", "US English not supported, using system default.")
                    tts?.setLanguage(Locale.getDefault())
                }

                ready = true

                // Speak any messages that were queued while we were waiting
                synchronized(pendingMessages) {
                    for (msg in pendingMessages) {
                        tts?.speak(msg, TextToSpeech.QUEUE_ADD, null, "msg_$msg")
                    }
                    pendingMessages.clear()
                }
            } else {
                Log.e("Speaker", "TTS Initialization failed")
            }
        }
    }

    fun say(text: String, flush: Boolean = true) {
        if (ready) {
            // QUEUE_FLUSH interrupts current speech, QUEUE_ADD waits its turn
            val queueMode = if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            tts?.speak(text, queueMode, null, "msg_$text")
        } else {
            // If not ready yet, save it for later
            synchronized(pendingMessages) {
                if (flush) pendingMessages.clear() // respect flush even in queue
                pendingMessages.add(text)
            }
        }
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
    }
}