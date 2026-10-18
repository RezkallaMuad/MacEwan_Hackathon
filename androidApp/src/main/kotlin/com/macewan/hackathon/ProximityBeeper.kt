package com.macewan.hackathon

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.SystemClock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin

class ProximityBeeper {
    private var job: Job? = null

    /** 0f = dead center of target, 1f = far away, null = no hand or object found */
    @Volatile var distance: Float? = null

    /** When true (app in background / screen off), no beeps. */
    @Volatile var muted = false

    /** Time constant of the distance smoothing. Higher = steadier but slower to react. */
    private val SMOOTHING_MS = 250f

    private val sampleRate = 22050
    private val beepSamples = (sampleRate * 0.06).toInt() // 60 ms beep

    private val track: AudioTrack = run {
        val pcm = ShortArray(beepSamples) { i ->
            val fade = minOf(1f, i / 200f, (beepSamples - i) / 200f) // avoids clicks
            (sin(2.0 * PI * 880.0 * i / sampleRate) * Short.MAX_VALUE * fade)
                .toInt().toShort()
        }
        AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcm.size * 2)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
            .also { it.write(pcm, 0, pcm.size) }
    }

    private fun beep(volume: Float) {
        track.stop()
        track.setVolume(volume.coerceIn(0f, 1f))
        track.reloadStaticData()
        track.play()
    }

    fun start(scope: CoroutineScope) {
        job?.cancel()
        job = scope.launch(Dispatchers.Default) {
            var smoothed = 1f
            var lastValidTime = 0L
            var lastTick = SystemClock.uptimeMillis()
            // Progress toward the next beep: 0 = just beeped, 1 = beep now.
            // Advancing a phase (instead of "time since last beep >= interval") means a
            // sudden change in distance smoothly speeds up / slows down the rhythm rather
            // than firing a beep immediately, which is what made it crackle like a Geiger counter.
            var phase = 1f

            while (isActive) {
                val now = SystemClock.uptimeMillis()
                val dt = (now - lastTick).coerceIn(0L, 100L).toFloat()
                lastTick = now

                if (muted) {
                    smoothed = 1f
                    lastValidTime = 0L
                    phase = 1f
                    delay(50)
                    continue
                }
                val d = distance

                // Time-based low-pass filter (same feel regardless of loop timing).
                val alpha = 1f - exp(-dt / SMOOTHING_MS)
                if (d != null) {
                    lastValidTime = now
                    smoothed += (d.coerceIn(0f, 1f) - smoothed) * alpha
                } else if (now - lastValidTime > 500) {
                    // Grace period: hold distance for up to 500ms during frame dropouts
                    smoothed += (1f - smoothed) * alpha
                }

                val hasActiveTarget = (d != null) || (now - lastValidTime <= 500)
                val closeness = (1f - smoothed).coerceIn(0f, 1f)
                val eased = closeness.toDouble().pow(1.1).toFloat()
                val intervalMs = 1000f - eased * 880f // 1000 ms far -> 120 ms close
                val volume = 0.15f + 0.85f * eased

                if (!hasActiveTarget) {
                    phase = 1f // beep right away when a target appears
                } else {
                    phase += dt / intervalMs
                    if (phase >= 1f) {
                        beep(volume)
                        phase = 0f
                    }
                }

                delay(16) // ~60 Hz loop
            }
        }
    }

    fun stop() {
        job?.cancel()
        track.release()
    }
}