package com.acoustichddsentinel.dsp

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Test jednostkowy weryfikujący poprawność cyfrowego przetwarzania sygnału (DSP)
 * pod kątem niezmienników ADR-001 (wymiary, brak NaN/Inf, Z-score).
 */
class AudioPreprocessorTest {

    @Test
    fun testAudioPreprocessorOutputDimensionsAndValues() {
        val preprocessor = AudioPreprocessor()

        // Wygenerowanie syntetycznego sygnału testowego (szum + sinusoida 1000 Hz, 66150 próbek)
        val testPcm = FloatArray(DspConstants.CHUNK_SAMPLES)
        for (i in testPcm.indices) {
            val t = i.toDouble() / DspConstants.SAMPLE_RATE
            testPcm[i] = (0.2 * Math.sin(2.0 * Math.PI * 1000.0 * t) + 0.05 * Math.random()).toFloat()
        }

        val flatTensor = preprocessor.processAudioChunk(testPcm)

        // 1. Weryfikacja wymiarów wyjściowych (3 * 128 * 259 = 99456)
        assertEquals(
            "Rozmiar tensora musi być dokładnie równy DspConstants.TOTAL_TENSOR_ELEMENTS",
            DspConstants.TOTAL_TENSOR_ELEMENTS,
            flatTensor.size
        )

        // 2. Weryfikacja braku wartości NaN oraz Inf
        for (i in flatTensor.indices) {
            val v = flatTensor[i]
            assertFalse("Wykryto wartość NaN na pozycji $i", v.isNaN())
            assertFalse("Wykryto wartość Nieskończoności (Inf) na pozycji $i", v.isInfinite())
        }

        // 3. Weryfikacja normalizacji Z-Score dla każdego z 3 kanałów
        val channelSize = DspConstants.N_MELS * DspConstants.FRAMES
        for (c in 0 until DspConstants.CHANNELS) {
            val offset = c * channelSize
            var sum = 0.0
            for (i in 0 until channelSize) {
                sum += flatTensor[offset + i]
            }
            val mean = (sum / channelSize).toFloat()

            var varSum = 0.0
            for (i in 0 until channelSize) {
                val diff = flatTensor[offset + i] - mean
                varSum += diff * diff
            }
            val std = sqrt(varSum / channelSize).toFloat()

            // Średnia powinna być bliska 0.0 (tolerancja 0.05), a odchylenie bliskie 1.0 (tolerancja 0.05)
            assertTrue("Średnia kanału $c powinna wynosić ~0.0 (jest $mean)", abs(mean) < 0.05f)
            assertTrue("Odchylenie kanału $c powinno wynosić ~1.0 (jest $std)", abs(std - 1.0f) < 0.05f)
        }
    }
}
