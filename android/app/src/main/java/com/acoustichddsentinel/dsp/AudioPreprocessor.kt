package com.acoustichddsentinel.dsp

import kotlin.math.*

/**
 * Główny procesor sygnału audio na urządzeniu mobilnym.
 * Implementuje 100% niezmienników DSP zdefiniowanych w ADR-001:
 * 1. Peak Normalization fali dźwiękowej (do zakresu [-1.0, 1.0]).
 * 2. Transformata STFT z oknem Hanna i paddingiem zwierciadlanym.
 * 3. Log-Mel Spektrogram (Kanał 0: 128 filtrów Mel, pasmo 50-10000 Hz, kompresja dB 10*log10).
 * 4. Delta - 1. pochodna czasowa (Kanał 1: detekcja impulsów Click of Death).
 * 5. Delta-Delta - 2. pochodna czasowa (Kanał 2: dynamika przyspieszeń kątowych).
 * 6. Z-Score Normalization per kanał (mean=0, std=1 z epsilonem 1e-6).
 *
 * Wynik: spłaszczony bufor float (3 * 128 * 259 = 99456 elementów)
 * gotowy do podania na wejście modelu ONNX (1, 3, 128, 259).
 */
class AudioPreprocessor(
    private val filterBank: MelFilterBank = MelFilterBank(),
    private val stft: StftTransformer = StftTransformer()
) {
    companion object {
        private const val DELTA_WIDTH = 9
    }

    /**
     * Główny potok przetwarzania surowego wycinka 3.0s audio.
     */
    fun processAudioChunk(rawPcm: FloatArray): FloatArray {
        // 1. Sprawdzenie i dopasowanie długości (wymagane 66150 próbek)
        val audio = FloatArray(DspConstants.CHUNK_SAMPLES)
        val copyLen = min(rawPcm.size, DspConstants.CHUNK_SAMPLES)
        System.arraycopy(rawPcm, 0, audio, 0, copyLen)

        // 2. Krok 1: Peak Normalization fali
        normalizeWaveformPeak(audio)

        // 3. Krok 2: Transformata STFT -> Spektrogram mocy [frames][freqBins]
        val powerSpec = stft.computePowerSpectrogram(audio)
        val numFrames = min(powerSpec.size, DspConstants.FRAMES)

        // 4. Krok 3: Przejście przez bank filtrów Mel -> [nMels][frames]
        val melSpec = Array(DspConstants.N_MELS) { FloatArray(numFrames) }
        val weights = filterBank.weights

        for (m in 0 until DspConstants.N_MELS) {
            val melFilter = weights[m]
            for (f in 0 until numFrames) {
                var sum = 0.0f
                val framePower = powerSpec[f]
                for (k in melFilter.indices) {
                    sum += melFilter[k] * framePower[k]
                }
                melSpec[m][f] = sum
            }
        }

        // 5. Krok 4: Kompresja decybelowa Log-Mel (Kanał 0): 10 * log10(max(spec, 1e-10))
        val channel0LogMel = Array(DspConstants.N_MELS) { FloatArray(DspConstants.FRAMES) }
        for (m in 0 until DspConstants.N_MELS) {
            for (f in 0 until numFrames) {
                val valClamped = max(melSpec[m][f], 1e-10f)
                channel0LogMel[m][f] = 10.0f * log10(valClamped)
            }
            // Uzupełnienie ewentualnych brakujących ramek do stałej liczby 259
            for (f in numFrames until DspConstants.FRAMES) {
                channel0LogMel[m][f] = channel0LogMel[m][numFrames - 1]
            }
        }

        // 6. Krok 5: Delta (Kanał 1) - 1. pochodna w czasie
        val channel1Delta = computeDelta(channel0LogMel, DELTA_WIDTH)

        // 7. Krok 6: Delta-Delta (Kanał 2) - 2. pochodna w czasie
        val channel2DeltaDelta = computeDelta(channel1Delta, DELTA_WIDTH)

        // 8. Krok 7: Z-score Normalization per kanał
        normalizeZScore(channel0LogMel)
        normalizeZScore(channel1Delta)
        normalizeZScore(channel2DeltaDelta)

        // 9. Spłaszczenie do tensora w formacie Channel-First (3, 128, 259)
        val flatTensor = FloatArray(DspConstants.TOTAL_TENSOR_ELEMENTS)
        val channelStride = DspConstants.N_MELS * DspConstants.FRAMES

        copyChannelToFlat(channel0LogMel, flatTensor, 0 * channelStride)
        copyChannelToFlat(channel1Delta, flatTensor, 1 * channelStride)
        copyChannelToFlat(channel2DeltaDelta, flatTensor, 2 * channelStride)

        return flatTensor
    }

    /**
     * Normalizacja szczytowa fali: dzielenie przez max(|audio|) z clipem do [-1.0, 1.0].
     */
    private fun normalizeWaveformPeak(audio: FloatArray) {
        var maxPeak = 0.0f
        for (sample in audio) {
            val absVal = abs(sample)
            if (absVal > maxPeak) {
                maxPeak = absVal
            }
        }

        if (maxPeak > 1e-6f) {
            for (i in audio.indices) {
                audio[i] = max(-1.0f, min(1.0f, audio[i] / maxPeak))
            }
        }
    }

    /**
     * Obliczenie współczynników Delta wzdłuż osi czasu (frames) o zadanym oknie N (width=9).
     * Zgodne z librosa.feature.delta(data, width=9, mode='edge').
     */
    private fun computeDelta(matrix: Array<FloatArray>, width: Int): Array<FloatArray> {
        val rows = matrix.size
        val cols = matrix[0].size
        val delta = Array(rows) { FloatArray(cols) }

        val halfWidth = width / 2
        var denom = 0.0f
        for (i in 1..halfWidth) {
            denom += 2.0f * (i * i)
        }

        for (r in 0 until rows) {
            val row = matrix[r]
            for (t in 0 until cols) {
                var num = 0.0f
                for (k in -halfWidth..halfWidth) {
                    if (k == 0) continue
                    val idx = min(max(t + k, 0), cols - 1) // mode='edge'
                    num += k * row[idx]
                }
                delta[r][t] = num / denom
            }
        }

        return delta
    }

    /**
     * Normalizacja Z-Score danej macierzy (kanału): (x - mean) / (std + 1e-6).
     */
    private fun normalizeZScore(matrix: Array<FloatArray>) {
        val rows = matrix.size
        val cols = matrix[0].size
        val total = (rows * cols).toDouble()

        var sum = 0.0
        for (r in 0 until rows) {
            val row = matrix[r]
            for (c in 0 until cols) {
                sum += row[c]
            }
        }
        val mean = (sum / total).toFloat()

        var varianceSum = 0.0
        for (r in 0 until rows) {
            val row = matrix[r]
            for (c in 0 until cols) {
                val diff = row[c] - mean
                varianceSum += diff * diff
            }
        }
        val std = sqrt(varianceSum / total).toFloat()

        val scale = 1.0f / (std + 1e-6f)
        for (r in 0 until rows) {
            val row = matrix[r]
            for (c in 0 until cols) {
                row[c] = (row[c] - mean) * scale
            }
        }
    }

    private fun copyChannelToFlat(matrix: Array<FloatArray>, target: FloatArray, offset: Int) {
        var cursor = offset
        for (r in matrix.indices) {
            val row = matrix[r]
            System.arraycopy(row, 0, target, cursor, row.size)
            cursor += row.size
        }
    }
}
