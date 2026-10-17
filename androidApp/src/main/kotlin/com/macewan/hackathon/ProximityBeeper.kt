package com.macewan.hackathon

import android.media.AudioManager
import android.media.ToneGenerator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class ProximityBeeper {
    private val tone = ToneGenerator(AudioManager.STREAM_MUSIC, 100)
    private var job: Job? = null

    /** 0f = touching, 1f = far away, null = no hand or object found */
    @Volatile var distance: Float? = null

    fun start(scope: CoroutineScope) {
        job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                val d = distance
                if (d == null) { delay(100); continue }
                tone.startTone(ToneGenerator.TONE_PROP_BEEP, 50)
                // far = 900 ms between beeps, touching = 90 ms (near-continuous)
                delay((90 + d.coerceIn(0f, 1f) * 810).toLong())
            }
        }
    }

    fun stop() { job?.cancel(); tone.release() }
}