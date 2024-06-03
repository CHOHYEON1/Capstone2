package com.example.mysoundclassification

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.location.Location
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.Menu
import android.view.MenuInflater
import android.view.MenuItem
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.BitmapDescriptorFactory
import com.google.android.gms.maps.model.Marker
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt


class MainActivity : AppCompatActivity(), OnMapReadyCallback {
    private val TAG = "MainActivity"
    private val REQUEST_RECORD_AUDIO = 1337
    private val LOCATION_PERMISSION_REQUEST_CODE = 1234

    private val modelPath = "lite-model_yamnet_classification_tflite_1.tflite"
    private val probabilityThreshold: Float = 0.3f

    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationCallback: LocationCallback
    private var currentLocation: Location? = null

    private lateinit var directionTextView: TextView
    private var audioRecord2: AudioRecord? = null
    private val sampleRate2 = 44100
    private val channelConfig2 = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat2 = AudioFormat.ENCODING_PCM_16BIT
    private var isRecording2 = false
    private val thresholdDb = 50

    private lateinit var mMap: GoogleMap
    private var isMapReady: Boolean = false

    private lateinit var textView: TextView
    private lateinit var recorderSpecsTextView: TextView
    private lateinit var vibrator: Vibrator

    private var timer: Timer? = null
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // Initialize the map
        val mapFragment = supportFragmentManager.findFragmentById(R.id.map) as SupportMapFragment
        mapFragment.getMapAsync(this)

        textView = findViewById(R.id.output)
        recorderSpecsTextView = findViewById(R.id.textViewAudioRecorderSpecs)
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        directionTextView = findViewById(R.id.directionTextView)

        requestPermissions()
    }

    override fun onMapReady(googleMap: GoogleMap) {
        mMap = googleMap
        isMapReady = true

        // 위치 권한 확인
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            // 권한이 허용된 경우 지도 초기화
            initializeMap()
        } else {
            // 권한이 없는 경우 권한 요청
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), LOCATION_PERMISSION_REQUEST_CODE)
        }
    }

    private fun initializeMap() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        mMap.isMyLocationEnabled = true

        // 위치 요청 설정
        val locationRequest = LocationRequest.create().apply {
            interval = 5000 // 5초 간격으로 위치 업데이트
            fastestInterval = 2000 // 2초 간격으로 위치 업데이트
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        }

        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                locationResult.lastLocation?.let {
                    currentLocation = it
                    val currentLatLng = LatLng(it.latitude, it.longitude)
                    mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(currentLatLng, 20f))
                }
            }
        }

        fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper())
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

        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initializeMap()
        }
    }

    private fun startAudioClassificationService() {
        val intent = Intent(this, AudioClassificationService::class.java)
        ContextCompat.startForegroundService(this, intent)

        startAudioClassification()
    }

    private var currentDirection: String = ""
    private var selectedSounds: Set<String> = emptySet()
    private val detectionTargets = listOf("dog", "honk", "siren", "bird")

    private fun startAudioClassification() {
        selectedSounds = loadSelectedSoundsFromSharedPreferences()
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate2, channelConfig2, audioFormat2)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return
        }

        audioRecord2 = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate2,
            channelConfig2,
            audioFormat2,
            minBufferSize
        )
        audioRecord2?.startRecording()
        isRecording2 = true

        Thread {
            while (isRecording2) {
                val buffer = ShortArray(minBufferSize)
                audioRecord2?.read(buffer, 0, buffer.size)

                val rmsSum = buffer.map { it.toDouble().pow(2) }.sum()
                val rms = sqrt(rmsSum / buffer.size)
                val db = rmsToDb(rms)

                if (db > thresholdDb) {
                    currentDirection = if (buffer.take(buffer.size / 2).sum() > buffer.drop(buffer.size / 2).sum()) "Left" else "Right"
                    runOnUiThread {
                        directionTextView.text = currentDirection
                    }
                }
            }
        }.start()

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
                    .joinToString(separator = "\n") { "${it.label}" }

                val detectedSound = detectionTargets.find { target ->
                    filteredModelOutput.any { it.label.contains(target, ignoreCase = true) }
                }

                if (outputStr.isNotEmpty()) {
                    runOnUiThread {
                        textView.text = outputStr
                        if (isTargetSoundDetected(outputStr, selectedSounds)) {
                            addDirectionalMarker(currentDirection, outputStr)
                            if (detectedSound != null) {
                                sendNotification(detectedSound)
                            }
                        } else {
                            Log.d("MainActivity", "Detected sound not in selected sounds: $outputStr")
                        }
                    }
                } else {
                    Log.d("MainActivity", "No sounds detected")
                }
            }
        }
    }

    private var lastSoundDetectionTimeMap = mutableMapOf<String, Long>()

    private fun isTargetSoundDetected(outputStr: String, selectedSounds: Set<String>): Boolean {
        val currentTime = System.currentTimeMillis()
        for (sound in selectedSounds) {
            if (outputStr.contains(sound, ignoreCase = true)) {
                // 해당 소리가 최근 5초 내에 감지되었는지 확인
                if (currentTime - (lastSoundDetectionTimeMap[sound] ?: 0) > 5000) {
                    // 최근 5초 내에 감지되지 않았으면 해당 소리 감지로 처리하고, 시간 기록
                    lastSoundDetectionTimeMap[sound] = currentTime
                    return true
                }
            }
        }
        return false
    }




    private fun vibrate(target: String) {
        if (vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val vibrationEffect = when (target) {
                    "siren" -> VibrationEffect.createWaveform(longArrayOf(0, 500, 200, 500, 200, 500), -1) // 진동 3번
                    "dog" -> VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE) // 진동 1번
                    "honk" -> VibrationEffect.createWaveform(longArrayOf(0, 500, 200, 500), -1) // 진동 2번
                    else -> null
                }
                vibrationEffect?.let { vibrator.vibrate(it) }
            } else {
                when (target) {
                    "siren" -> {
                        vibrator.vibrate(longArrayOf(0, 500, 200, 500, 200, 500), -1) // 진동 3번
                    }
                    "dog" -> {
                        vibrator.vibrate(500) // 진동 1번
                    }
                    "honk" -> {
                        vibrator.vibrate(longArrayOf(0, 500, 200, 500), -1) // 진동 2번
                    }
                }
            }
        }
    }

    private var notificationId = 2 // Start with a unique ID
    private fun sendNotification(detectedSound: String) {
        val channelId = "AudioClassificationServiceChannel"
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channelName = "Audio Classification Service"
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(channelId, channelName, importance)
            notificationManager.createNotificationChannel(channel)
        }

        // 기본 진동 비활성화
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Sound Detected")
            .setContentText("Detected: $detectedSound")
            .setSmallIcon(R.drawable.ic_launcher_background)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setVibrate(longArrayOf(0L)) // 빈 진동 패턴 설정
            .setAutoCancel(true)
            .build()

        notificationManager.notify(notificationId++, notification)

        // 맞춤 진동 패턴 적용
        vibrate(detectedSound)
    }



    private val markerList = mutableListOf<Marker>()

    private fun getResizedBitmap(@DrawableRes drawableRes: Int, width: Int, height: Int): Bitmap? {
        val imageBitmap = BitmapFactory.decodeResource(resources, drawableRes)
        return Bitmap.createScaledBitmap(imageBitmap, width, height, false)
    }

    private fun addDirectionalMarker(direction: String, outputStr: String) {
        currentLocation?.let { location ->
            val currentLatLng = LatLng(location.latitude, location.longitude)
            val distance = 0.0001
            val angle = Math.toRadians(location.bearing.toDouble())

            val newLatLng = when (direction) {
                "Left" -> LatLng(
                    currentLatLng.latitude + distance * sin(angle + Math.PI / 2),
                    currentLatLng.longitude + distance * cos(angle + Math.PI / 2)
                )
                "Right" -> LatLng(
                    currentLatLng.latitude + distance * sin(angle - Math.PI / 2),
                    currentLatLng.longitude + distance * cos(angle - Math.PI / 2)
                )
                else -> currentLatLng
            }

            val markerOptions = MarkerOptions()
                .position(newLatLng)
                .title(direction)

            val markerImage = when {
                outputStr.contains("dog", ignoreCase = true) || outputStr.contains("bark", ignoreCase = true) -> R.drawable.dog
                outputStr.contains("honk", ignoreCase = true) || outputStr.contains("horn", ignoreCase = true) -> R.drawable.honk
                outputStr.contains("siren", ignoreCase = true) -> R.drawable.siren
                outputStr.contains("bird", ignoreCase = true) -> R.drawable.bird
                else -> R.drawable.accessibility
            }

            val resizedMarkerImage = getResizedBitmap(markerImage, 200, 200)
            resizedMarkerImage?.let {
                markerOptions.icon(BitmapDescriptorFactory.fromBitmap(it))
            }

            val marker = mMap.addMarker(markerOptions)
            marker?.let { safeMarker ->
                markerList.add(safeMarker)

                handler.postDelayed({
                    safeMarker.remove()
                    markerList.remove(safeMarker)
                }, 5000)
            }
        }
    }

    private fun rmsToDb(rms: Double): Double {
        return 20 * log10(rms)
    }

    override fun onDestroy() {
        super.onDestroy()
        isRecording2 = false
        audioRecord2?.stop()
        audioRecord2?.release()
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        val inflater: MenuInflater = menuInflater
        inflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                openSettings()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun openSettings() {
        val intent = Intent(this, SettingsActivity::class.java)
        startActivity(intent)
    }

    private fun loadSelectedSoundsFromSharedPreferences(): Set<String> {
        val sharedPrefs = getSharedPreferences("SoundPreferences", Context.MODE_PRIVATE)
        Log.d("MainActivity", "SharedPrefs content: ${sharedPrefs.all}")

        val selectedSoundsSet = mutableSetOf<String>()
        if (sharedPrefs.getBoolean("dog", false)) selectedSoundsSet.add("dog")
        if (sharedPrefs.getBoolean("honk", false)) selectedSoundsSet.add("honk")
        if (sharedPrefs.getBoolean("siren", false)) selectedSoundsSet.add("siren")
        if (sharedPrefs.getBoolean("bird", false)) selectedSoundsSet.add("bird")

        Log.d("MainActivity", "Loaded preferences: $selectedSoundsSet")

        return selectedSoundsSet
    }


    override fun onResume() {
        super.onResume()

        selectedSounds = loadSelectedSoundsFromSharedPreferences()

        if (selectedSounds.isNotEmpty()) {
            textView.text = "Selected Sounds: ${selectedSounds.joinToString(", ")}"
            Log.d("MainActivity", "Selected sounds: ${selectedSounds.joinToString(", ")}")
        } else {
            textView.text = "No sounds selected"
        }

        resetAudioClassification()
    }

    private fun resetAudioClassification() {
        timer?.cancel()
        isRecording2 = false
        audioRecord2?.stop()
        audioRecord2?.release()

        startAudioClassification()
    }
}
