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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

import androidx.compose.ui.platform.LocalContext


//holds current camera alignment
enum class TiltDirection { PERFECT, TILT_DOWN, TILT_UP }

data class AlignmentState(
    val pitch: Float = 0f,
    val isAligned: Boolean = false,
    val direction: TiltDirection = TiltDirection.TILT_DOWN
)

//composable hook to track when the phone should buzz
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

        // arrays to hold matrix math
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
                // user enters the sweet spot
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

                // user leaves the sweet spot single low buzz
                if (!isCurrentlyAligned && wasAligned) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        vibrator.vibrate(VibrationEffect.createOneShot(40L, VibrationEffect.DEFAULT_AMPLITUDE))
                    } else {
                        @Suppress("DEPRECATION")
                        vibrator.vibrate(40L)
                    }
                }

                wasAligned = isCurrentlyAligned

                // update the ui compose state
                alignmentState = AlignmentState(
                    pitch = pitchDeg,
                    isAligned = isCurrentlyAligned
                )
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                // dont really need this
            }
        }

        // register sensor listener
        sensorManager.registerListener(
            listener,
            rotationSensor,
            SensorManager.SENSOR_DELAY_UI
        )

        // clean up screen close
        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }

    return alignmentState
}