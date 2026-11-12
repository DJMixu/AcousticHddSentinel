package com.acoustichddsentinel.ml;

import java.util.Collections;
import java.util.Map;

/**
 * Wynik klasyfikacji akustycznej pojedynczego lub zagregowanego badania HDD.
 */
public final class ClassificationResult {

    private final String dominantClass;
    private final float confidence;
    private final Map<String, Float> probabilities;
    private final boolean isAnomaly;
    private final long inferenceTimeMs;
    private final String recommendation;
    private final String detectedRpm;

    public ClassificationResult(String dominantClass, float confidence, Map<String, Float> probabilities,
                                boolean isAnomaly, long inferenceTimeMs) {
        this(dominantClass, confidence, probabilities, isAnomaly, inferenceTimeMs, null, null);
    }

    public ClassificationResult(String dominantClass, float confidence, Map<String, Float> probabilities,
                                boolean isAnomaly, long inferenceTimeMs, String detectedRpm, String customRecommendation) {
        this.dominantClass = dominantClass;
        this.confidence = confidence;
        this.probabilities = probabilities != null ? Collections.unmodifiableMap(probabilities) : Collections.emptyMap();
        this.isAnomaly = isAnomaly;
        this.inferenceTimeMs = inferenceTimeMs;
        this.detectedRpm = detectedRpm;
        this.recommendation = customRecommendation != null ? customRecommendation : generateRecommendation(dominantClass, isAnomaly);
    }

    public String getDominantClass() {
        return dominantClass;
    }

    public float getConfidence() {
        return confidence;
    }

    public Map<String, Float> getProbabilities() {
        return probabilities;
    }

    public boolean isAnomaly() {
        return isAnomaly;
    }

    public long getInferenceTimeMs() {
        return inferenceTimeMs;
    }

    public String getRecommendation() {
        return recommendation;
    }

    public String getDetectedRpm() {
        return detectedRpm;
    }

    public String getFormattedConfidence() {
        return String.format(java.util.Locale.US, "%.1f%%", confidence * 100.0f);
    }

    public static String generateRecommendation(String predictedClass, boolean isAnomaly) {
        if (!isAnomaly || "normal".equalsIgnoreCase(predictedClass)) {
            return "Dysk sprawny. Profil akustyczny łożyska i głowic w normie.";
        }
        switch (predictedClass.toLowerCase()) {
            case "clicking":
                return "UWAGA: Wykryto powtarzalne uderzenia pozycjonera (Click of Death). Natychmiast zabezpiecz dane i odłącz zasilanie nośnika!";
            case "beeping":
                return "KRYTYCZNY: Dźwięk zablokowania lub zatarcia silnika wrzeciona (Seized Spindle). Nie uruchamiaj dysku, ryzyko zniszczenia talerzy!";
            case "scratching":
                return "ALARM: Wykryto mechaniczne tarcie głowic o powierzchnię magnetyczną (Head Crash). Ryzyko nieodwracalnej utraty danych!";
            default:
                return "Wykryto anomalię mechaniczną dysku HDD. Zalecana ostrożność i weryfikacja SMART.";
        }
    }
}
