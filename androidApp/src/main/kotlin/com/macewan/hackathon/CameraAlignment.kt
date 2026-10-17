package com.macewan.hackathon

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Holds the current alignment state of the camera.
 */
enum class TiltDirection { PERFECT, TILT_DOWN, TILT_UP }

data class AlignmentState(
    val pitch: Float = 0f,
    val isAligned: Boolean = false,
    val direction: TiltDirection = TiltDirection.TILT_DOWN
)

/**
 * Composable hook to track phone tilt and buzz when aligned.
 */
@Composable
fun rememberCameraAlignment(): AlignmentState {
    val context = LocalContext.current
    var alignmentState by remember { mutableStateOf(AlignmentState()) }

    // System services
    val sensorManager = remember {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }
    val vibrator = remember {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    DisposableEffect(Unit) {
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        var wasAligned = false

        // Arrays to hold matrix math
        val rotationMatrix = FloatArray(9)
        val orientationAngles = FloatArray(3)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event == null) return

                //calculate pitch to degrees
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                SensorManager.getOrientation(rotationMatrix, orientationAngles)

                //convert angle to radians
                val pitchRad = orientationAngles[1]
                val pitchDeg = kotlin.math.abs(Math.toDegrees(pitchRad.toDouble()).toFloat())

                //sweet spot between 35 and 55 degrees
                val direction = when {
                    pitchDeg < 35f -> TiltDirection.TILT_DOWN
                    pitchDeg > 55f -> TiltDirection.TILT_UP
                    else -> TiltDirection.PERFECT
                }
                val isCurrentlyAligned = (direction == TiltDirection.PERFECT)


                //trigger the haptics on the loading
                // A. User just ENTERED the sweet spot -> Double Buzz ("Lock on!")
                if (isCurrentlyAligned && !wasAligned) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        // Timing pattern: [delay, vibrate, pause, vibrate]
                        val timings = longArrayOf(0, 70, 80, 70)
                        vibrator.vibrate(VibrationEffect.createWaveform(timings, -1))
                    } else {
                        @Suppress("DEPRECATION")
                        vibrator.vibrate(100L)
                    }
                }

// B. User just SLIPPED OUT of the sweet spot -> Single Warning Buzz ("Lost lock!")
                if (!isCurrentlyAligned && wasAligned) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        vibrator.vibrate(VibrationEffect.createOneShot(40L, VibrationEffect.DEFAULT_AMPLITUDE))
                    } else {
                        @Suppress("DEPRECATION")
                        vibrator.vibrate(40L)
                    }
                }

                wasAligned = isCurrentlyAligned

                // Update Compose UI state
                alignmentState = AlignmentState(
                    pitch = pitchDeg,
                    isAligned = isCurrentlyAligned
                )
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                // Not needed for this
            }
        }

        // Register sensor listener
        sensorManager.registerListener(
            listener,
            rotationSensor,
            SensorManager.SENSOR_DELAY_UI
        )

        // Cleanup when screen closes
        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }

    return alignmentState
}

/**
 * Optional ready-to-use visual badge to display on screen during debugging.
 */
@Composable
fun AlignmentBadge(
    state: AlignmentState,
    modifier: Modifier = Modifier
) {
    val backgroundColor = if (state.isAligned) Color(0xFF2E7D32) else Color(0xFFC62828)
    val text = if (state.isAligned) {
        "Angle: ${state.pitch.toInt()}° (ALIGNED)"
    } else {
        "Angle: ${state.pitch.toInt()}° (TILT TO 45°)"
    }

    Text(
        text = text,
        color = Color.White,
        fontSize = 12.sp,
        modifier = modifier
            .background(backgroundColor.copy(alpha = 0.85f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    )
}