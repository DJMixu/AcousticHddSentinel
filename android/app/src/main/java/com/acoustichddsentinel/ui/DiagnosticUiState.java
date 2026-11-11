package com.acoustichddsentinel.ui;

import android.graphics.Bitmap;

import com.acoustichddsentinel.data.DiagnosticModel;
import com.acoustichddsentinel.ml.ClassificationResult;

/**
 * Niezmienny stan interfejsu użytkownika w architekturze MVVM (UI State Pattern).
 */
public final class DiagnosticUiState {

    public enum Status {
        IDLE,
        RECORDING,
        PROCESSING,
        SUCCESS,
        ERROR
    }

    private final Status status;
    private final int recordingElapsedMs;
    private final int recordingTotalMs;
    private final float recordingRms;
    private final String statusMessage;
    private final DiagnosticModel completedDiagnostic;
    private final ClassificationResult classificationResult;
    private final Bitmap spectrogramBitmap;
    private final String errorMessage;

    private DiagnosticUiState(Status status, int recordingElapsedMs, int recordingTotalMs, float recordingRms,
                              String statusMessage, DiagnosticModel completedDiagnostic,
                              ClassificationResult classificationResult, Bitmap spectrogramBitmap, String errorMessage) {
        this.status = status;
        this.recordingElapsedMs = recordingElapsedMs;
        this.recordingTotalMs = recordingTotalMs;
        this.recordingRms = recordingRms;
        this.statusMessage = statusMessage;
        this.completedDiagnostic = completedDiagnostic;
        this.classificationResult = classificationResult;
        this.spectrogramBitmap = spectrogramBitmap;
        this.errorMessage = errorMessage;
    }

    public static DiagnosticUiState idle() {
        return new DiagnosticUiState(Status.IDLE, 0, 0, 0.0f, "Gotowy do badania", null, null, null, null);
    }

    public static DiagnosticUiState recording(int elapsedMs, int totalMs, float rms) {
        return new DiagnosticUiState(Status.RECORDING, elapsedMs, totalMs, rms,
            String.format(java.util.Locale.US, "Rejestracja... %d s / %d s (RMS: %.4f)", elapsedMs / 1000, totalMs / 1000, rms),
            null, null, null, null);
    }

    public static DiagnosticUiState processing(String message) {
        return new DiagnosticUiState(Status.PROCESSING, 0, 0, 0.0f, message, null, null, null, null);
    }

    public static DiagnosticUiState success(DiagnosticModel diagnostic, ClassificationResult result, Bitmap bitmap) {
        return new DiagnosticUiState(Status.SUCCESS, 0, 0, 0.0f, "Badanie zakończone pomyślnie",
            diagnostic, result, bitmap, null);
    }

    public static DiagnosticUiState error(String errorMessage) {
        return new DiagnosticUiState(Status.ERROR, 0, 0, 0.0f, errorMessage, null, null, null, errorMessage);
    }

    public Status getStatus() {
        return status;
    }

    public int getRecordingElapsedMs() {
        return recordingElapsedMs;
    }

    public int getRecordingTotalMs() {
        return recordingTotalMs;
    }

    public float getRecordingRms() {
        return recordingRms;
    }

    public String getStatusMessage() {
        return statusMessage;
    }

    public DiagnosticModel getCompletedDiagnostic() {
        return completedDiagnostic;
    }

    public ClassificationResult getClassificationResult() {
        return classificationResult;
    }

    public Bitmap getSpectrogramBitmap() {
        return spectrogramBitmap;
    }

    public String getErrorMessage() {
        return errorMessage;
    }
}
