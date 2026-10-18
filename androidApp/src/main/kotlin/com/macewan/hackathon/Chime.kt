package com.macewan.hackathon

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin

/** Gentle UI sounds: a soft "still loading" chime and a rising "ready" tune. */
class Chime {
    private val sampleRate = 22050

    /** Soft two-note chime (E5 -> A5), played periodically while loading. */
    private val pending = track(
        render(listOf(659.25 to 0.0, 880.0 to 0.18), noteSec = 0.6, volume = 0.35)
    )

    /** Rising three-note tune (C5 E5 G5 C6), played when ready. */
    private val ready = track(
        render(listOf(523.25 to 0.0, 659.25 to 0.12, 783.99 to 0.24, 1046.5 to 0.36), noteSec = 0.7, volume = 0.45)
    )

    fun pending() = play(pending)
    fun ready() = play(ready)

    fun release() {
        pending.release()
        ready.release()
    }

    private fun play(t: AudioTrack) {
        try {
            t.stop()
            t.reloadStaticData()
            t.play()
        } catch (_: IllegalStateException) { }
    }

    /** Mixes bell-like notes (freq, start time in s) with a quick attack and soft decay. */
    private fun render(notes: List<Pair<Double, Double>>, noteSec: Double, volume: Double): ShortArray {
        val total = ((notes.maxOf { it.second } + noteSec) * sampleRate).toInt()
        val mix = DoubleArray(total)
        for ((freq, start) in notes) {
            val s0 = (start * sampleRate).toInt()
            val n = (noteSec * sampleRate).toInt()
            for (i in 0 until n) {
                val idx = s0 + i
                if (idx >= total) break
                val t = i.toDouble() / sampleRate
                val env = minOf(1.0, t / 0.01) * exp(-t * 6.0)   // 10 ms attack, soft decay
                // Fundamental plus a quiet octave for a warmer, bell-like tone
                val v = sin(2 * PI * freq * t) + 0.25 * sin(2 * PI * freq * 2 * t)
                mix[idx] += v * env
            }
        }
        val peak = mix.maxOf { kotlin.math.abs(it) }.coerceAtLeast(1e-6)
        return ShortArray(total) { (mix[it] / peak * volume * Short.MAX_VALUE).toInt().toShort() }
    }

    private fun track(pcm: ShortArray): AudioTrack =
        AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
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
