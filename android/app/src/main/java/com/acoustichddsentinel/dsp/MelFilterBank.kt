package com.acoustichddsentinel.dsp

import kotlin.math.*

/**
 * Generator wag banku filtrów Mel (Mel Filterbank) zgodny z biblioteką Librosa (norm='slaney').
 * Tworzy macierz o wymiarach (n_mels, n_fft / 2 + 1).
 */
class MelFilterBank(
    private val nMels: Int = DspConstants.N_MELS,
    private val nFft: Int = DspConstants.N_FFT,
    private val sampleRate: Int = DspConstants.SAMPLE_RATE,
    private val fMin: Float = DspConstants.FMIN,
    private val fMax: Float = DspConstants.FMAX
) {
    private val nFftBins = nFft / 2 + 1
    // Tablica wag o wymiarach: weights[melIndex][freqBinIndex]
    val weights: Array<FloatArray> = Array(nMels) { FloatArray(nFftBins) }

    init {
        computeFilterBank()
    }

    private fun hzToMel(hz: Float): Float {
        // Formuła zgodna ze skalą Mel (HTK / Slaney)
        return 2595.0f * log10(1.0f + hz / 700.0f)
    }

    private fun melToHz(mel: Float): Float {
        return 700.0f * (10.0f.pow(mel / 2595.0f) - 1.0f)
    }

    private fun computeFilterBank() {
        val minMel = hzToMel(fMin)
        val maxMel = hzToMel(fMax)

        // Równomierne rozmieszczenie punktów w skali Mel
        val melPoints = FloatArray(nMels + 2)
        val melStep = (maxMel - minMel) / (nMels + 1)
        for (i in melPoints.indices) {
            melPoints[i] = minMel + i * melStep
        }

        // Konwersja punktów Mel z powrotem na częstotliwości w Hz
        val hzPoints = FloatArray(nMels + 2)
        for (i in hzPoints.indices) {
            hzPoints[i] = melToHz(melPoints[i])
        }

        // Przyporządkowanie punktów Hz do indeksów koszyków FFT (bins)
        val binStep = sampleRate.toFloat() / nFft.toFloat()
        val binPoints = FloatArray(nMels + 2)
        for (i in binPoints.indices) {
            binPoints[i] = hzPoints[i] / binStep
        }

        // Konstrukcja trójkątnych filtrów pasmowych
        for (m in 0 until nMels) {
            val left = binPoints[m]
            val center = binPoints[m + 1]
            val right = binPoints[m + 2]

            for (k in 0 until nFftBins) {
                val weight = when {
                    k < left -> 0.0f
                    k <= center -> {
                        if (center - left > 0.0001f) (k - left) / (center - left) else 0.0f
                    }
                    k <= right -> {
                        if (right - center > 0.0001f) (right - k) / (right - center) else 0.0f
                    }
                    else -> 0.0f
                }
                weights[m][k] = weight
            }

            // Normalizacja powierzchni trójkąta (Slaney-style area normalization)
            val enorm = 2.0f / (hzPoints[m + 2] - hzPoints[m])
            for (k in 0 until nFftBins) {
                weights[m][k] *= enorm
            }
        }
    }
}
