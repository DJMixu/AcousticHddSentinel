package com.acoustichddsentinel.data;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Model danych pojedynczego wpisu diagnostycznego w bazie SQLite.
 * Rozszerzony o szczegółowe parametry akustyczne, ścieżkę do nagrania WAV,
 * pełny rozkład prawdopodobieństw klas oraz rekomendacje naprawcze.
 */
public final class DiagnosticModel {

    private long id;
    private final String hddName;
    private final long timestamp;
    private final boolean hasAnomaly;
    private final String spectrogramPath;
    private final String predictedClass;
    private final float confidence;

    // Nowe pola wzbogacone w wersji bazy v2:
    private String audioPath;
    private float probBeeping;
    private float probClicking;
    private float probNormal;
    private float probScratching;
    private float crestFactor;
    private float peakRms;
    private float scanDurationS;
    private String recommendation;

    /**
     * Pełny konstruktor dla odczytu z bazy SQLite (wersja v2).
     */
    public DiagnosticModel(long id, String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence,
                           String audioPath, float probBeeping, float probClicking,
                           float probNormal, float probScratching, float crestFactor,
                           float peakRms, float scanDurationS, String recommendation) {
        this.id = id;
        this.hddName = hddName;
        this.timestamp = timestamp;
        this.hasAnomaly = hasAnomaly;
        this.spectrogramPath = spectrogramPath;
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.audioPath = audioPath;
        this.probBeeping = probBeeping;
        this.probClicking = probClicking;
        this.probNormal = probNormal;
        this.probScratching = probScratching;
        this.crestFactor = crestFactor;
        this.peakRms = peakRms;
        this.scanDurationS = scanDurationS;
        this.recommendation = recommendation;
    }

    /**
     * Konstruktor dla nowego zapisu do bazy bez znanego jeszcze ID.
     */
    public DiagnosticModel(String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence,
                           String audioPath, float probBeeping, float probClicking,
                           float probNormal, float probScratching, float crestFactor,
                           float peakRms, float scanDurationS, String recommendation) {
        this(0, hddName, timestamp, hasAnomaly, spectrogramPath, predictedClass, confidence,
             audioPath, probBeeping, probClicking, probNormal, probScratching,
             crestFactor, peakRms, scanDurationS, recommendation);
    }

    /**
     * Konstruktor kompatybilności wstecznej z wersją v1 (z id).
     */
    public DiagnosticModel(long id, String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence) {
        this(id, hddName, timestamp, hasAnomaly, spectrogramPath, predictedClass, confidence,
             null, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, null);
    }

    /**
     * Konstruktor kompatybilności wstecznej z wersją v1 (bez id).
     */
    public DiagnosticModel(String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence) {
        this(0, hddName, timestamp, hasAnomaly, spectrogramPath, predictedClass, confidence,
             null, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, null);
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getHddName() {
        return hddName;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public boolean hasAnomaly() {
        return hasAnomaly;
    }

    public String getSpectrogramPath() {
        return spectrogramPath;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public float getConfidence() {
        return confidence;
    }

    public String getAudioPath() {
        return audioPath;
    }

    public void setAudioPath(String audioPath) {
        this.audioPath = audioPath;
    }

    public float getProbBeeping() {
        return probBeeping;
    }

    public void setProbBeeping(float probBeeping) {
        this.probBeeping = probBeeping;
    }

    public float getProbClicking() {
        return probClicking;
    }

    public void setProbClicking(float probClicking) {
        this.probClicking = probClicking;
    }

    public float getProbNormal() {
        return probNormal;
    }

    public void setProbNormal(float probNormal) {
        this.probNormal = probNormal;
    }

    public float getProbScratching() {
        return probScratching;
    }

    public void setProbScratching(float probScratching) {
        this.probScratching = probScratching;
    }

    public float getCrestFactor() {
        return crestFactor;
    }

    public void setCrestFactor(float crestFactor) {
        this.crestFactor = crestFactor;
    }

    public float getPeakRms() {
        return peakRms;
    }

    public void setPeakRms(float peakRms) {
        this.peakRms = peakRms;
    }

    public float getScanDurationS() {
        return scanDurationS;
    }

    public void setScanDurationS(float scanDurationS) {
        this.scanDurationS = scanDurationS;
    }

    public String getRecommendation() {
        return recommendation;
    }

    public void setRecommendation(String recommendation) {
        this.recommendation = recommendation;
    }

    public String getFormattedDate() {
        SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.getDefault());
        return sdf.format(new Date(timestamp));
    }
}
