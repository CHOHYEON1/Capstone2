package com.example.mysoundclassification

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val REQUEST_RECORD_AUDIO = 1337

    private val modelPath = "lite-model_yamnet_classification_tflite_1.tflite"
    private val probabilityThreshold: Float = 0.3f

    private lateinit var textView: TextView
    private lateinit var recorderSpecsTextView: TextView
    private lateinit var imageViewDog: ImageView
    private lateinit var imageViewBark: ImageView
    private lateinit var imageViewHonk: ImageView
    private lateinit var imageViewFireAlarm: ImageView
    private lateinit var imageViewSiren: ImageView
    private lateinit var imageViewVehicleHorn: ImageView
    private lateinit var vibrator: Vibrator

    private var timer: Timer? = null
    private val handler = Handler(Looper.getMainLooper())
    private var currentRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textView = findViewById(R.id.output)
        recorderSpecsTextView = findViewById(R.id.textViewAudioRecorderSpecs)
        imageViewDog = findViewById(R.id.imageViewDog)
        imageViewBark = findViewById(R.id.imageViewBark)
        imageViewHonk = findViewById(R.id.imageViewHonk)
        imageViewFireAlarm = findViewById(R.id.imageViewFireAlarm)
        imageViewSiren = findViewById(R.id.imageViewSiren)
        imageViewVehicleHorn = findViewById(R.id.imageViewVehicleHorn)
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        requestPermissions()
    }

    private fun requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        } else {
            startAudioClassificationService()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startAudioClassificationService()
        } else {
            textView.text = "Audio recording permission denied"
        }
    }

    private fun startAudioClassificationService() {
        val intent = Intent(this, AudioClassificationService::class.java)
        ContextCompat.startForegroundService(this, intent)

        startAudioClassification()
    }

    private fun startAudioClassification() {
        val classifier = AudioClassifier.createFromFile(this, modelPath)
        val tensor = classifier.createInputTensorAudio()
        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs = "Number Of Channels: ${format.channels}\nSample Rate: ${format.sampleRate}"
        recorderSpecsTextView.text = recorderSpecs

        val record = classifier.createAudioRecord()
        record.startRecording()

        timer = Timer().apply {
            scheduleAtFixedRate(1, 750) {
                val numberOfSamples = tensor.load(record)
                val output = classifier.classify(tensor)

                val filteredModelOutput = output[0].categories.filter {
                    it.score > probabilityThreshold
                }

                val outputStr = filteredModelOutput.sortedBy { -it.score }
                    .joinToString(separator = "\n") { "${it.label} -> ${it.score} " }

                if (outputStr.isNotEmpty()) {
                    runOnUiThread {
                        textView.text = outputStr
                        if (isTargetSoundDetected(outputStr)) {
                            showRelevantImage(outputStr)
                        }
                    }
                }
            }
        }
    }

    private fun isTargetSoundDetected(outputStr: String): Boolean {
        return outputStr.contains("dog", ignoreCase = true) ||
                outputStr.contains("bark", ignoreCase = true) ||
                outputStr.contains("honk", ignoreCase = true) ||
                outputStr.contains("fire alarm", ignoreCase = true) ||
                outputStr.contains("siren", ignoreCase = true) ||
                outputStr.contains("vehicle horn", ignoreCase = true)
    }

    private fun vibrate() {
        if (vibrator.hasVibrator()) {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                vibrator.vibrate(500)
            }
        }
    }

    private fun showRelevantImage(outputStr: String) {
        hideAllImages()

        val imageViewToShow = when {
            outputStr.contains("dog", ignoreCase = true) -> imageViewDog
            outputStr.contains("bark", ignoreCase = true) -> imageViewBark
            outputStr.contains("honk", ignoreCase = true) -> imageViewHonk
            outputStr.contains("fire alarm", ignoreCase = true) -> imageViewFireAlarm
            outputStr.contains("siren", ignoreCase = true) -> imageViewSiren
            outputStr.contains("vehicle horn", ignoreCase = true) -> imageViewVehicleHorn
            else -> null
        }

        imageViewToShow?.let {
            it.visibility = ImageView.VISIBLE
            vibrate()

            currentRunnable?.let { handler.removeCallbacks(it) }
            currentRunnable = Runnable {
                it.visibility = ImageView.GONE
            }.also { runnable ->
                handler.postDelayed(runnable, 3000)
            }
        }
    }

    private fun hideAllImages() {
        imageViewDog.visibility = ImageView.GONE
        imageViewBark.visibility = ImageView.GONE
        imageViewHonk.visibility = ImageView.GONE
        imageViewFireAlarm.visibility = ImageView.GONE
        imageViewSiren.visibility = ImageView.GONE
        imageViewVehicleHorn.visibility = ImageView.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        timer?.cancel()
    }
}
