package com.macewan.hackathon

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.sin

class ProximityBeeper {
    private var job: Job? = null

    /** 0f = dead center of target, 1f = far away, null = no hand or object found */
    @Volatile var distance: Float? = null

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
            while (isActive) {
                val d = distance
                if (d == null) {
                    smoothed = 1f
                    delay(100)
                    continue
                }
                // smooth out hand-tracking jitter
                smoothed += (d.coerceIn(0f, 1f) - smoothed) * 0.35f

                val closeness = 1f - smoothed            // 0 far, 1 center
                val eased = Math.pow(closeness.toDouble(), 1.4).toFloat()   // between linear (1.0) and squared (2.0)
                val intervalMs = (1000 - eased * 880).toLong()              // 1000 ms far -> 120 ms center   // 1200 ms far -> 120 ms center
                val volume = 0.15f + 0.85f * eased       // quiet far, loud at center

                beep(volume)
                delay(intervalMs)
            }
        }
    }

    fun stop() {
        job?.cancel()
        track.release()
    }
}