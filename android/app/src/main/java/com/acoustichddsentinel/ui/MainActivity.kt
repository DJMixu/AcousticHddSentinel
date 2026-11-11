package com.acoustichddsentinel.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.acoustichddsentinel.AcousticSentinelApp
import com.acoustichddsentinel.R
import com.acoustichddsentinel.databinding.ActivityMainBinding
import com.acoustichddsentinel.ml.AcousticClassifier
import com.acoustichddsentinel.ml.ClassificationResult
import com.acoustichddsentinel.ui.viewmodel.DiagnosticUiState
import com.acoustichddsentinel.ui.viewmodel.MainViewModel
import com.acoustichddsentinel.ui.viewmodel.MainViewModelFactory
import kotlinx.coroutines.launch

/**
 * Główna aktywność aplikacji Acoustic HDD Sentinel.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var historyAdapter: DiagnosticHistoryAdapter

    private val viewModel: MainViewModel by viewModels {
        val app = application as AcousticSentinelApp
        val classifier = AcousticClassifier(this)
        MainViewModelFactory(app.repository, classifier)
    }

    private val requestAudioPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            triggerDiagnosis()
        } else {
            Toast.makeText(this, "Wymagane uprawnienie do mikrofonu, aby przeprowadzić diagnozę.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        observeState()
    }

    private fun setupUI() {
        historyAdapter = DiagnosticHistoryAdapter { id ->
            viewModel.deleteHistoryItem(id)
        }
        binding.rvHistory.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = historyAdapter
        }

        binding.btnStartDiagnosis.setOnClickListener {
            checkPermissionAndStart()
        }

        binding.btnClearHistory.setOnClickListener {
            viewModel.clearAllHistory()
        }
    }

    private fun checkPermissionAndStart() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> {
                triggerDiagnosis()
            }
            else -> {
                requestAudioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun triggerDiagnosis() {
        val currentState = viewModel.uiState.value
        if (currentState is DiagnosticUiState.Recording) {
            viewModel.stopDiagnosis()
        } else {
            val hddName = binding.etHddName.text.toString()
            viewModel.startDiagnosis(hddName)
        }
    }

    private fun observeState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.uiState.collect { state ->
                        renderState(state)
                    }
                }
                launch {
                    viewModel.history.collect { items ->
                        historyAdapter.submitList(items)
                        binding.tvHistoryEmpty.visibility = if (items.isEmpty()) View.VISIBLE else View.GONE
                    }
                }
            }
        }
    }

    private fun renderState(state: DiagnosticUiState) {
        when (state) {
            is DiagnosticUiState.Idle -> {
                binding.btnStartDiagnosis.text = getString(R.string.btn_start_diagnosis)
                binding.btnStartDiagnosis.isEnabled = true
                binding.progressBarAudio.visibility = View.GONE
                binding.tvRecordingStatus.text = getString(R.string.status_idle)
            }
            is DiagnosticUiState.Recording -> {
                binding.btnStartDiagnosis.text = getString(R.string.btn_stop_diagnosis)
                binding.btnStartDiagnosis.isEnabled = true
                binding.progressBarAudio.visibility = View.VISIBLE
                val progressPercent = ((state.durationMs / 3000.0) * 100).toInt().coerceIn(0, 100)
                binding.progressBarAudio.progress = progressPercent
                binding.tvRecordingStatus.text = getString(R.string.status_recording_fmt, state.durationMs / 1000f, state.currentRms)
            }
            is DiagnosticUiState.Analyzing -> {
                binding.btnStartDiagnosis.text = getString(R.string.status_analyzing)
                binding.btnStartDiagnosis.isEnabled = false
                binding.progressBarAudio.visibility = View.VISIBLE
                binding.progressBarAudio.isIndeterminate = true
                binding.tvRecordingStatus.text = getString(R.string.status_running_inference)
            }
            is DiagnosticUiState.ResultReady -> {
                binding.btnStartDiagnosis.text = getString(R.string.btn_start_diagnosis)
                binding.btnStartDiagnosis.isEnabled = true
                binding.progressBarAudio.visibility = View.GONE
                binding.progressBarAudio.isIndeterminate = false
                binding.tvRecordingStatus.text = getString(R.string.status_ready)
                displayResult(state.result, state.hddName)
            }
            is DiagnosticUiState.Error -> {
                binding.btnStartDiagnosis.text = getString(R.string.btn_start_diagnosis)
                binding.btnStartDiagnosis.isEnabled = true
                binding.progressBarAudio.visibility = View.GONE
                binding.tvRecordingStatus.text = "Błąd"
                Toast.makeText(this, state.message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun displayResult(result: ClassificationResult, hddName: String) {
        binding.cardResult.visibility = View.VISIBLE
        binding.tvResultHddName.text = hddName
        binding.tvResultClass.text = result.predictedClass.uppercase()
        binding.tvResultConfidence.text = String.format("%.1f%% pewności (czas: %d ms)", result.confidence * 100f, result.inferenceTimeMs)
        binding.tvResultRecommendation.text = result.diagnosticMessage

        if (result.isHealthy) {
            binding.tvResultBadge.text = "SPRAWNY"
            binding.tvResultBadge.setBackgroundResource(R.drawable.bg_badge_normal)
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_normal_text))
            binding.ivResultIcon.setImageResource(R.drawable.ic_check_circle)
        } else {
            binding.tvResultBadge.text = "AWARIA"
            binding.tvResultBadge.setBackgroundResource(R.drawable.bg_badge_anomaly)
            binding.tvResultBadge.setTextColor(ContextCompat.getColor(this, R.color.status_danger_text))
            binding.ivResultIcon.setImageResource(R.drawable.ic_warning)
        }

        // Aktualizacja pasków prawdopodobieństwa
        val probs = result.probabilities
        val pNormal = ((probs["normal"] ?: 0f) * 100).toInt()
        val pClicking = ((probs["clicking"] ?: 0f) * 100).toInt()
        val pBeeping = ((probs["beeping"] ?: 0f) * 100).toInt()
        val pScratching = ((probs["scratching"] ?: 0f) * 100).toInt()

        binding.progressNormal.progress = pNormal
        binding.tvProbNormal.text = "$pNormal%"

        binding.progressClicking.progress = pClicking
        binding.tvProbClicking.text = "$pClicking%"

        binding.progressBeeping.progress = pBeeping
        binding.tvProbBeeping.text = "$pBeeping%"

        binding.progressScratching.progress = pScratching
        binding.tvProbScratching.text = "$pScratching%"
    }
}
