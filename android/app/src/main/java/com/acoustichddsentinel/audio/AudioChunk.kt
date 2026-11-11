package com.acoustichddsentinel.audio

/**
 * Zarejestrowany 3-sekundowy wycinek sygnału audio z mikrofonu.
 */
data class AudioChunk(
    val samples: FloatArray,
    val sampleRate: Int,
    val rms: Float,
    val timestampMs: Long = System.currentTimeMillis()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AudioChunk
        return samples.contentEquals(other.samples) && sampleRate == other.sampleRate
    }

    override fun hashCode(): Int {
        var result = samples.contentHashCode()
        result = 31 * result + sampleRate
        return result
    }
}
