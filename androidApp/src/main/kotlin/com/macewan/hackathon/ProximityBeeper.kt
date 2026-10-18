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

    @Volatile var distance: Float? = null
    @Volatile var muted = false

    // REDUCED from 250ms to 70ms: 3.5x faster reaction to rapid hand moves!
    private val SMOOTHING_MS = 70f

    private val sampleRate = 22050
    // Shortened to 30ms so rapid-fire clicks don't overlap or buffer-choke
    private val beepSamples = (sampleRate * 0.030).toInt()

    private val track: AudioTrack = run {
        val pcm = ShortArray(beepSamples) { i ->
            val fade = minOf(1f, i / 100f, (beepSamples - i) / 100f) // avoids clicks
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

                val alpha = 1f - exp(-dt / SMOOTHING_MS)
                if (d != null) {
                    lastValidTime = now
                    val target = d.coerceIn(0f, 1f)
                    // If we are touching / very close (< 0.15), snap immediately with no delay!
                    if (target < 0.15f) {
                        smoothed = target
                    } else {
                        smoothed += (target - smoothed) * alpha
                    }
                } else if (now - lastValidTime > 300) {
                    smoothed += (1f - smoothed) * alpha
                }

                val hasActiveTarget = (d != null) || (now - lastValidTime <= 300)
                val closeness = (1f - smoothed).coerceIn(0f, 1f)

                // pow(0.75) ramps up earlier in the reaching path
                val eased = closeness.toDouble().pow(0.75).toFloat()

                // Fast range: 800ms interval (far) down to 50ms (touching = rapid geiger clicks!)
                val intervalMs = 800f - (eased * 750f)
                val volume = 0.2f + 0.8f * eased

                if (!hasActiveTarget) {
                    phase = 1f
                } else {
                    phase += dt / intervalMs
                    if (phase >= 1f) {
                        beep(volume)
                        phase = 0f
                    }
                }

                delay(12) // ~80 Hz check loop for instant response
            }
        }
    }

    fun stop() {
        job?.cancel()
        track.release()
    }
}