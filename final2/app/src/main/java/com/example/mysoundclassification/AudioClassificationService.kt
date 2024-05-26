package com.example.mysoundclassification

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import androidx.core.app.NotificationCompat
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class AudioClassificationService : Service() {

    private lateinit var vibrator: Vibrator
    private lateinit var classifier: AudioClassifier
    private val probabilityThreshold: Float = 0.3f
    private val modelPath = "lite-model_yamnet_classification_tflite_1.tflite"
    private val detectionTargets = listOf("dog", "Bark", "Honk", "Fire alarm", "Siren", "Vehicle horn")
    private val channelId = "AudioClassificationServiceChannel"
    private var notificationId = 2 // Start with a unique ID
    private var lastDetectedSound: String? = null // Variable to store the last detected sound

    override fun onCreate() {
        super.onCreate()

        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        classifier = AudioClassifier.createFromFile(this, modelPath)
        val tensor = classifier.createInputTensorAudio()
        val record = classifier.createAudioRecord()
        record.startRecording()

        startForegroundService()

        Timer().scheduleAtFixedRate(0, 500) {
            val numberOfSamples = tensor.load(record)
            val output = classifier.classify(tensor)

            val filteredModelOutput = output[0].categories.filter {
                it.score > probabilityThreshold
            }

            val outputStr = filteredModelOutput.sortedBy { -it.score }
                .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

            val detectedSound = detectionTargets.find { target ->
                filteredModelOutput.any { it.label.equals(target, ignoreCase = true) }
            }

            if (detectedSound != null && detectedSound != lastDetectedSound) {
                lastDetectedSound = detectedSound
                vibrate()
                sendNotification(detectedSound)
            }
        }
    }

    private fun startForegroundService() {
        val channelName = "Audio Classification Service"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notificationChannel = NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_DEFAULT)
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(notificationChannel)
        }

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Audio Classification Service")
            .setContentText("Running audio classification in the background")
            .setSmallIcon(R.drawable.ic_launcher_foreground) // Ensure this icon exists
            .build()

        startForeground(1, notification)
    }

    private fun vibrate() {
        if (vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                vibrator.vibrate(500)
            }
        }
    }

    private fun sendNotification(detectedSound: String) {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        val notification = NotificationCompat.Builder(this, channelId)
            .setSmallIcon(R.drawable.ic_launcher_foreground) // Ensure this icon exists
            .setContentTitle("Sound Detected")
            .setContentText("Detected sound: $detectedSound")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        notificationManager.notify(notificationId++, notification)
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY // Ensure the service is restarted if it gets terminated
    }

    override fun onDestroy() {
        super.onDestroy()
        // 필요한 경우 리소스를 정리
    }
}
