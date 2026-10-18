package com.macewan.hackathon

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF

/** Color names the app can say and understand. */
val COLOR_WORDS = setOf(
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"
)
private val COLOR_SYNONYMS = mapOf("grey" to "gray", "violet" to "purple")

/** Rough everyday color name for an HSV value (h 0..360, s and v 0..1). */
fun colorName(h: Float, s: Float, v: Float): String = when {
    v < 0.2f -> "black"
    s < 0.18f -> when {
        v > 0.8f -> "white"
        v < 0.35f -> "black"
        else -> "gray"
    }
    (h < 40f || h >= 345f) && v < 0.5f -> "brown"   // dark reds/oranges look brown
    h < 15f || h >= 345f -> "red"
    h < 40f -> "orange"
    h < 70f -> "yellow"
    h < 165f -> "green"
    h < 255f -> "blue"
    h < 290f -> "purple"
    else -> "pink"
}

/**
 * Most common color in the middle half of [box] (pixel coords in [bmp]).
 * Only the middle is sampled so the background around the object doesn't leak in.
 */
fun dominantColor(bmp: Bitmap, box: RectF): String? {
    val l = (box.left + box.width() * 0.25f).coerceIn(0f, bmp.width - 1f)
    val r = (box.right - box.width() * 0.25f).coerceIn(0f, bmp.width - 1f)
    val t = (box.top + box.height() * 0.25f).coerceIn(0f, bmp.height - 1f)
    val b = (box.bottom - box.height() * 0.25f).coerceIn(0f, bmp.height - 1f)
    if (r <= l || b <= t) return null
    val counts = HashMap<String, Int>()
    val hsv = FloatArray(3)
    val n = 12
    for (iy in 0 until n) for (ix in 0 until n) {
        val x = (l + (r - l) * (ix + 0.5f) / n).toInt()
        val y = (t + (b - t) * (iy + 0.5f) / n).toInt()
        Color.colorToHSV(bmp.getPixel(x, y), hsv)
        val name = colorName(hsv[0], hsv[1], hsv[2])
        counts[name] = (counts[name] ?: 0) + 1
    }
    return counts.maxByOrNull { it.value }?.key
}

/** Color word said alongside [label] ("the red cup" -> "red"), or null. */
fun matchColor(spoken: String, label: String): String? {
    val words = spoken.lowercase().split(Regex("[^a-z]+")).filter { it.isNotEmpty() }
        .map { COLOR_SYNONYMS[it] ?: it }
    val colors = words.filter { it in COLOR_WORDS }.toMutableList()
    // "orange" is also a fruit: if that's the item, one "orange" is the item, not a color.
    if (label == "orange") colors.remove("orange")
    return colors.firstOrNull()
}
