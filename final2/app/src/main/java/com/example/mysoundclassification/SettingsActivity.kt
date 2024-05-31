package com.example.mysoundclassification

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    private lateinit var checkBoxDog: CheckBox
    private lateinit var checkBoxHonk: CheckBox
    private lateinit var checkBoxSiren: CheckBox
    private lateinit var checkBoxBird: CheckBox

    private lateinit var preferences: SharedPreferences

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        preferences = getSharedPreferences("SoundPreferences", Context.MODE_PRIVATE)

        checkBoxDog = findViewById(R.id.checkBoxDog)
        checkBoxHonk = findViewById(R.id.checkBoxHonk)
        checkBoxSiren = findViewById(R.id.checkBoxSiren)
        checkBoxBird = findViewById(R.id.checkBoxBird)

        loadPreferences()

        val buttonSave: Button = findViewById(R.id.buttonSave)
        buttonSave.setOnClickListener {
            savePreferences()
            finish()
        }
    }

    private fun loadPreferences() {
        checkBoxDog.isChecked = preferences.getBoolean("dog", false)
        checkBoxHonk.isChecked = preferences.getBoolean("honk", false)
        checkBoxSiren.isChecked = preferences.getBoolean("siren", false)
        checkBoxBird.isChecked = preferences.getBoolean("bird", false)
    }

    private fun savePreferences() {
        val editor = preferences.edit()
        editor.putBoolean("dog", checkBoxDog.isChecked)
        editor.putBoolean("honk", checkBoxHonk.isChecked)
        editor.putBoolean("siren", checkBoxSiren.isChecked)
        editor.putBoolean("bird", checkBoxBird.isChecked)
        editor.apply()

        Log.d("SettingsActivity", "Saved preferences: dog=${checkBoxDog.isChecked}, honk=${checkBoxHonk.isChecked}, siren=${checkBoxSiren.isChecked}, bird=${checkBoxBird.isChecked}")
    }

}
