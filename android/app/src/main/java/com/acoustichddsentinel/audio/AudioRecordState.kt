package com.acoustichddsentinel.audio

/**
 * Stan rejestratora audio w czasie rzeczywistym.
 */
sealed class AudioRecordState {
    object Idle : AudioRecordState()
    object Initializing : AudioRecordState()
    data class Recording(val durationMs: Long, val currentRms: Float) : AudioRecordState()
    data class Completed(val chunk: AudioChunk) : AudioRecordState()
    data class Error(val message: String) : AudioRecordState()
}
