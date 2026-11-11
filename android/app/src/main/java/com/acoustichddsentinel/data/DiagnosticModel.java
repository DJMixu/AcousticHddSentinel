package com.acoustichddsentinel.data;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Model danych pojedynczego wpisu diagnostycznego w bazie SQLite.
 */
public final class DiagnosticModel {

    private long id;
    private final String hddName;
    private final long timestamp;
    private final boolean hasAnomaly;
    private final String spectrogramPath;
    private final String predictedClass;
    private final float confidence;

    public DiagnosticModel(long id, String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence) {
        this.id = id;
        this.hddName = hddName;
        this.timestamp = timestamp;
        this.hasAnomaly = hasAnomaly;
        this.spectrogramPath = spectrogramPath;
        this.predictedClass = predictedClass;
        this.confidence = confidence;
    }

    public DiagnosticModel(String hddName, long timestamp, boolean hasAnomaly,
                           String spectrogramPath, String predictedClass, float confidence) {
        this(0, hddName, timestamp, hasAnomaly, spectrogramPath, predictedClass, confidence);
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

    public String getFormattedDate() {
        SimpleDateFormat sdf = new SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.getDefault());
        return sdf.format(new Date(timestamp));
    }
}
