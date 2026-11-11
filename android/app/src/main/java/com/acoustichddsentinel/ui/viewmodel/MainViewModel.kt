package com.acoustichddsentinel.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.acoustichddsentinel.audio.AudioRecorder
import com.acoustichddsentinel.audio.AudioRecordState
import com.acoustichddsentinel.data.local.DiagnosticEntity
import com.acoustichddsentinel.data.repository.DiagnosticRepository
import com.acoustichddsentinel.dsp.AudioPreprocessor
import com.acoustichddsentinel.dsp.DspConstants
import com.acoustichddsentinel.ml.AcousticClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ViewModel zarządzający logiką nagrywania, przetwarzania sygnału i inferencji ML.
 */
class MainViewModel(
    private val repository: DiagnosticRepository,
    private val classifier: AcousticClassifier,
    private val preprocessor: AudioPreprocessor = AudioPreprocessor(),
    private val audioRecorder: AudioRecorder = AudioRecorder()
) : ViewModel() {

    private val _uiState = MutableStateFlow<DiagnosticUiState>(DiagnosticUiState.Idle)
    val uiState: StateFlow<DiagnosticUiState> = _uiState.asStateFlow()

    val history: StateFlow<List<DiagnosticEntity>> = repository.allDiagnostics
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    init {
        // Obserwacja stanu rejestratora audio
        viewModelScope.launch {
            audioRecorder.recordingState.collect { state ->
                when (state) {
                    is AudioRecordState.Recording -> {
                        _uiState.value = DiagnosticUiState.Recording(state.durationMs, state.currentRms)
                    }
                    is AudioRecordState.Error -> {
                        _uiState.value = DiagnosticUiState.Error(state.message)
                    }
                    else -> Unit
                }
            }
        }
    }

    /**
     * Rozpoczyna procedurę diagnozy akustycznej podanego dysku.
     */
    fun startDiagnosis(hddName: String) {
        val sanitizedHddName = if (hddName.isBlank()) "Nieznany HDD" else hddName.trim()

        audioRecorder.startRecording { chunk ->
            // Rejestracja zakończona, przechodzimy do analizy DSP + ONNX
            viewModelScope.launch {
                _uiState.value = DiagnosticUiState.Analyzing

                // Sprawdzenie bramki ciszy (RMS < 0.002)
                if (chunk.rms < DspConstants.RMS_SILENCE_THRESHOLD) {
                    _uiState.value = DiagnosticUiState.Error(
                        "Zbyt niski poziom sygnału (RMS = ${String.format("%.5f", chunk.rms)} < ${DspConstants.RMS_SILENCE_THRESHOLD}). Zbliż mikrofon do obudowy dysku."
                    )
                    return@launch
                }

                try {
                    // Wykonanie DSP i inferencji w tle (Dispatchers.Default)
                    val result = withContext(Dispatchers.Default) {
                        val flatTensor = preprocessor.processAudioChunk(chunk.samples)
                        classifier.classify(flatTensor)
                    }

                    // Zapis wyniku w lokalnej bazie SQLite (Room)
                    val entity = DiagnosticEntity(
                        hddName = sanitizedHddName,
                        timestamp = System.currentTimeMillis(),
                        predictedClass = result.predictedClass,
                        confidence = result.confidence,
                        hasAnomaly = result.isAnomaly
                    )
                    repository.saveDiagnostic(entity)

                    _uiState.value = DiagnosticUiState.ResultReady(result, sanitizedHddName)
                } catch (e: Exception) {
                    _uiState.value = DiagnosticUiState.Error("Błąd analizy akustycznej: ${e.message}")
                }
            }
        }
    }

    fun stopDiagnosis() {
        audioRecorder.stopRecording()
        _uiState.value = DiagnosticUiState.Idle
    }

    fun deleteHistoryItem(id: Long) {
        viewModelScope.launch {
            repository.deleteDiagnostic(id)
        }
    }

    fun clearAllHistory() {
        viewModelScope.launch {
            repository.clearHistory()
        }
    }

    override fun onCleared() {
        super.onCleared()
        audioRecorder.stopRecording()
        classifier.close()
    }
}
