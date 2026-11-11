package com.acoustichddsentinel.ml

/**
 * Wynik klasyfikacji akustycznej dysku HDD.
 */
data class ClassificationResult(
    val predictedClass: String,
    val confidence: Float,
    val probabilities: Map<String, Float>,
    val isAnomaly: Boolean,
    val inferenceTimeMs: Long
) {
    val isHealthy: Boolean get() = predictedClass.equals("normal", ignoreCase = true)

    val diagnosticMessage: String
        get() = when (predictedClass.lowercase()) {
            "normal" -> "Dysk sprawny. Parametry akustyczne łożyska i głowic w normie."
            "clicking" -> "Krytyczna awaria: Click of Death! Uderzanie głowic o ogranicznik lub awaria pozycjonera servo."
            "beeping" -> "Krytyczna awaria: Zablokowane wrzeciono silnika lub przylepienie głowic do talerza (stiction)."
            "scratching" -> "KATASTROFA: Fizyczne tarcie głowic o talerz (head crash)! Natychmiast odłącz zasilanie!"
            else -> "Nieznany stan akustyczny: $predictedClass"
        }
}
