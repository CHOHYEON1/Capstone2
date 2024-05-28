package com.example.mysoundclassification

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
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
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.GoogleMap
import com.google.android.gms.maps.OnMapReadyCallback
import com.google.android.gms.maps.SupportMapFragment
import com.google.android.gms.maps.model.LatLng
import com.google.android.gms.maps.model.MarkerOptions
import com.google.android.gms.maps.model.BitmapDescriptorFactory
import com.google.android.gms.maps.model.BitmapDescriptor
import com.google.android.gms.maps.model.Marker
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), OnMapReadyCallback {
    private val TAG = "MainActivity"
    private val REQUEST_RECORD_AUDIO = 1337
    private val LOCATION_PERMISSION_REQUEST_CODE = 1234

    private val modelPath = "lite-model_yamnet_classification_tflite_1.tflite"
    private val probabilityThreshold: Float = 0.3f

    private lateinit var locationManager: LocationManager
    private var currentLocation: Location? = null

    private lateinit var directionTextView: TextView
    private var audioRecord2: AudioRecord? = null
    private val sampleRate2 = 44100
    private val channelConfig2 = AudioFormat.CHANNEL_IN_STEREO
    private val audioFormat2 = AudioFormat.ENCODING_PCM_16BIT
    private var isRecording2 = false
    private val thresholdDb = 50  // 임계값 설정 (데시벨)

    private lateinit var mMap: GoogleMap
    private var isMapReady: Boolean = false

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

        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        // Initialize the map
        val mapFragment = supportFragmentManager.findFragmentById(R.id.map) as SupportMapFragment
        mapFragment.getMapAsync(this)

        textView = findViewById(R.id.output)
        recorderSpecsTextView = findViewById(R.id.textViewAudioRecorderSpecs)
        imageViewDog = findViewById(R.id.imageViewDog)
        imageViewBark = findViewById(R.id.imageViewBark)
        imageViewHonk = findViewById(R.id.imageViewHonk)
        imageViewFireAlarm = findViewById(R.id.imageViewFireAlarm)
        imageViewSiren = findViewById(R.id.imageViewSiren)
        imageViewVehicleHorn = findViewById(R.id.imageViewVehicleHorn)
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        directionTextView = findViewById(R.id.directionTextView)

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO
            )
        } else {
            startAudioProcessing()
        }

        requestPermissions()
    }

    override fun onMapReady(googleMap: GoogleMap) {
        mMap = googleMap
        isMapReady = true

        // 위치 권한 확인
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED) {
            // 권한이 허용된 경우 지도 초기화
            initializeMap()
        } else {
            // 권한이 없는 경우 권한 요청
            ActivityCompat.requestPermissions(this,
                arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION),
                LOCATION_PERMISSION_REQUEST_CODE)
        }
    }

    private fun initializeMap() {
        // 지도 설정
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        mMap.isMyLocationEnabled = true

        // 현재 위치 가져오기
        val location = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
        location?.let {
            val currentLatLng = LatLng(it.latitude, it.longitude)
            mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(currentLatLng, 21f))
        }
    }

    private fun startAudioProcessing() {
        val minBufferSize =
            AudioRecord.getMinBufferSize(sampleRate2, channelConfig2, audioFormat2)
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
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

                val leftChannelSum = mutableListOf<Double>()
                val rightChannelSum = mutableListOf<Double>()

                for (i in buffer.indices step 2) {
                    leftChannelSum.add(buffer[i].toDouble().pow(2))
                    rightChannelSum.add(buffer[i + 1].toDouble().pow(2))
                }

                val leftRms = sqrt(leftChannelSum.average())
                val rightRms = sqrt(rightChannelSum.average())

                val leftDb = rmsToDb(leftRms)
                val rightDb = rmsToDb(rightRms)

                if (leftDb > thresholdDb || rightDb > thresholdDb) {
                    val direction = if (leftDb > rightDb) "Left" else "Right"
                    runOnUiThread {
                        directionTextView.text = direction
                    }
                }
            }
        }.start()
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
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty()&& grantResults[0] == PackageManager.PERMISSION_GRANTED) {
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

    private fun rmsToDb(rms: Double): Double {
        return 20 * log10(rms)
    }

    override fun onDestroy() {
        super.onDestroy()
        isRecording2 = false
        audioRecord2?.stop()
        audioRecord2?.release()
    }
}

