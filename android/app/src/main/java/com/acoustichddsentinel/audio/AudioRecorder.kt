package com.acoustichddsentinel.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.acoustichddsentinel.dsp.DspConstants
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.sqrt

/**
 * Menedżer nagrywania audio z mikrofonu urządzenia za pomocą AudioRecord.
 * Konfiguracja: 22050 Hz, 16-bit PCM, Mono, 3.0s (66150 próbek).
 */
class AudioRecorder {
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _recordingState = MutableStateFlow<AudioRecordState>(AudioRecordState.Idle)
    val recordingState: StateFlow<AudioRecordState> = _recordingState.asStateFlow()

    @SuppressLint("MissingPermission")
    fun startRecording(onChunkReady: (AudioChunk) -> Unit) {
        if (_recordingState.value is AudioRecordState.Recording) return

        _recordingState.value = AudioRecordState.Initializing

        val sampleRate = DspConstants.SAMPLE_RATE
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT

        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            _recordingState.value = AudioRecordState.Error("Błąd inicjalizacji bufora mikrofonu.")
            return
        }

        val bufferSizeInBytes = maxOf(minBufferSize, DspConstants.CHUNK_SAMPLES * 2)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSizeInBytes
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                _recordingState.value = AudioRecordState.Error("Nie udało się zainicjalizować AudioRecord.")
                return
            }

            audioRecord?.startRecording()

            recordingJob = scope.launch {
                val totalSamplesNeeded = DspConstants.CHUNK_SAMPLES
                val recordedFloats = FloatArray(totalSamplesNeeded)
                val readBuffer = ShortArray(1024)
                var samplesCollected = 0
                val startTime = System.currentTimeMillis()

                while (isActive && samplesCollected < totalSamplesNeeded) {
                    val readCount = audioRecord?.read(readBuffer, 0, minOf(readBuffer.size, totalSamplesNeeded - samplesCollected)) ?: 0
                    if (readCount > 0) {
                        for (i in 0 until readCount) {
                            recordedFloats[samplesCollected + i] = readBuffer[i] / 32768.0f
                        }
                        samplesCollected += readCount

                        // Wyliczanie bieżącej energii RMS
                        val currentRms = computeRms(readBuffer, readCount)
                        val elapsedMs = System.currentTimeMillis() - startTime
                        _recordingState.value = AudioRecordState.Recording(elapsedMs, currentRms)
                    } else if (readCount < 0) {
                        _recordingState.value = AudioRecordState.Error("Błąd odczytu strumienia audio: $readCount")
                        break
                    }
                }

                if (samplesCollected >= totalSamplesNeeded) {
                    val fullRms = computeFloatRms(recordedFloats)
                    val chunk = AudioChunk(
                        samples = recordedFloats,
                        sampleRate = sampleRate,
                        rms = fullRms
                    )
                    _recordingState.value = AudioRecordState.Completed(chunk)
                    onChunkReady(chunk)
                }
            }
        } catch (e: Exception) {
            _recordingState.value = AudioRecordState.Error("Wyjątek podczas nagrywania: ${e.message}")
        }
    }

    fun stopRecording() {
        recordingJob?.cancel()
        recordingJob = null

        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {
        } finally {
            audioRecord = null
            _recordingState.value = AudioRecordState.Idle
        }
    }

    private fun computeRms(buffer: ShortArray, count: Int): Float {
        if (count == 0) return 0.0f
        var sumSquares = 0.0
        for (i in 0 until count) {
            val norm = buffer[i] / 32768.0
            sumSquares += norm * norm
        }
        return sqrt(sumSquares / count).toFloat()
    }

    private fun computeFloatRms(buffer: FloatArray): Float {
        if (buffer.isEmpty()) return 0.0f
        var sumSquares = 0.0
        for (sample in buffer) {
            sumSquares += (sample * sample).toDouble()
        }
        return sqrt(sumSquares / buffer.size).toFloat()
    }
}
