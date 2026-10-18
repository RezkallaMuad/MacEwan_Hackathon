package com.macewan.hackathon

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import kotlin.math.sqrt

/**
 * Listens to the phone shakes using the accelerometer
 *
 * @param enabled only enabled the shake when the user is in focus mode
 * @param shakeThresholdG g force for the shake, 1 is resting and above 2.5 is intentionally done
 * @param cooldownMs prevents multiple triggers when shaking the device
 * @param onShake call back is executed when the shake is detected
 */
@Composable
fun rememberShakeDetector(
    enabled: Boolean = true,
    shakeThresholdG: Float = 3.0f, //standard apparently
    cooldownMs: Long = 1200L,
    onShake: () -> Unit
) {
    val context = LocalContext.current

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

    DisposableEffect(enabled) {
        if (!enabled) {
            return@DisposableEffect onDispose {}
        }

        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        var lastShakeTimestamp = 0L

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event == null) return

                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]

                // gravity to 9.81m/s squared
                val gX = x / SensorManager.GRAVITY_EARTH
                val gY = y / SensorManager.GRAVITY_EARTH
                val gZ = z / SensorManager.GRAVITY_EARTH

                // total gforce magnitude
                val gForce = sqrt(gX * gX + gY * gY + gZ * gZ)

                val now = SystemClock.uptimeMillis()

                // check if the shake exceeds the threshold limit we set
                if (gForce > shakeThresholdG && (now - lastShakeTimestamp) > cooldownMs) {
                    lastShakeTimestamp = now

                    // make the phone buzz so the user can tell when the reset occurs
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        vibrator.vibrate(
                            VibrationEffect.createOneShot(120L, VibrationEffect.DEFAULT_AMPLITUDE)
                        )
                    } else {
                        @Suppress("DEPRECATION")
                        vibrator.vibrate(120L)
                    }

                    // fire the callback
                    onShake()
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
                // dont need
            }
        }

        sensorManager.registerListener(
            listener,
            accelerometer,
            SensorManager.SENSOR_DELAY_UI
        )

        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }
}