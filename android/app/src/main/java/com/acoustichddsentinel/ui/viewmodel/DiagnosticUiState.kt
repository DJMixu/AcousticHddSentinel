package com.acoustichddsentinel.ui.viewmodel

import com.acoustichddsentinel.ml.ClassificationResult

/**
 * Stan interfejsu użytkownika w trakcie procesu diagnostycznego.
 */
sealed class DiagnosticUiState {
    object Idle : DiagnosticUiState()
    data class Recording(val durationMs: Long, val currentRms: Float) : DiagnosticUiState()
    object Analyzing : DiagnosticUiState()
    data class ResultReady(val result: ClassificationResult, val hddName: String) : DiagnosticUiState()
    data class Error(val message: String) : DiagnosticUiState()
}
