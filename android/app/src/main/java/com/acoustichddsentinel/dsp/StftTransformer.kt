package com.acoustichddsentinel.dsp

import kotlin.math.*

/**
 * Moduł transformaty STFT (Short-Time Fourier Transform) z oknem Hanna
 * oraz paddingiem zwierciadlanym (reflect padding) zgodny z librosa.stft.
 */
class StftTransformer(
    private val nFft: Int = DspConstants.N_FFT,
    private val hopLength: Int = DspConstants.HOP_LENGTH
) {
    private val nFftBins = nFft / 2 + 1
    private val window: FloatArray = FloatArray(nFft)

    init {
        // Okno Hanna: w[n] = 0.5 * (1 - cos(2 * PI * n / N))
        for (i in 0 until nFft) {
            window[i] = (0.5 * (1.0 - cos(2.0 * PI * i.toDouble() / nFft.toDouble()))).toFloat()
        }
    }

    /**
     * Oblicza spektrogram mocy (Power Spectrogram) o wymiarach [frames][nFftBins].
     * Wejście: sygnał audio 1D o długości np. 66150 próbek.
     */
    fun computePowerSpectrogram(signal: FloatArray): Array<FloatArray> {
        val padLength = nFft / 2
        val paddedSignal = reflectPad(signal, padLength)

        val totalPadded = paddedSignal.size
        val numFrames = 1 + (totalPadded - nFft) / hopLength

        val powerSpec = Array(numFrames) { FloatArray(nFftBins) }
        val real = DoubleArray(nFft)
        val imag = DoubleArray(nFft)

        for (frameIdx in 0 until numFrames) {
            val start = frameIdx * hopLength
            for (i in 0 until nFft) {
                real[i] = (paddedSignal[start + i] * window[i]).toDouble()
                imag[i] = 0.0
            }

            fftRadix2(real, imag)

            for (k in 0 until nFftBins) {
                val r = real[k]
                val im = imag[k]
                powerSpec[frameIdx][k] = (r * r + im * im).toFloat()
            }
        }

        return powerSpec
    }

    /**
     * Dopełnienie sygnału zwierciadlanie na brzegach (reflect pad), jak w librosa.
     */
    private fun reflectPad(signal: FloatArray, pad: Int): FloatArray {
        val n = signal.size
        val padded = FloatArray(n + 2 * pad)

        // Środek
        System.arraycopy(signal, 0, padded, pad, n)

        // Lewa strona (odbicie bez powtarzania elementu brzegowego)
        for (i in 0 until pad) {
            padded[pad - 1 - i] = signal[min(i + 1, n - 1)]
        }

        // Prawa strona
        for (i in 0 until pad) {
            padded[pad + n + i] = signal[max(0, n - 2 - i)]
        }

        return padded
    }

    /**
     * Szybka transformata Fouriera (FFT Cooley-Tukey Radix-2 in-place).
     * Zakłada nFft = 1024 (potęga 2).
     */
    private fun fftRadix2(real: DoubleArray, imag: DoubleArray) {
        val n = real.size

        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val tempR = real[i]; real[i] = real[j]; real[j] = tempR
                val tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }

        // Cooley-Tukey
        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wstepR = cos(angle)
            val wstepI = sin(angle)
            val halfLen = len / 2

            var i = 0
            while (i < n) {
                var wR = 1.0
                var wI = 0.0
                for (k in 0 until halfLen) {
                    val uR = real[i + k]
                    val uI = imag[i + k]
                    val vR = real[i + k + halfLen] * wR - imag[i + k + halfLen] * wI
                    val vI = real[i + k + halfLen] * wI + imag[i + k + halfLen] * wR

                    real[i + k] = uR + vR
                    imag[i + k] = uI + vI
                    real[i + k + halfLen] = uR - vR
                    imag[i + k + halfLen] = uI - vI

                    val nextWR = wR * wstepR - wI * wstepI
                    wI = wR * wstepI + wI * wstepR
                    wR = nextWR
                }
                i += len
            }
            len *= 2
        }
    }
}
