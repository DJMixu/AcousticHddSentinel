package com.acoustichddsentinel.dsp;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe warstwy DSP i preprocesora akustycznego w czystej Javie.
 * Weryfikuje niezmienniki ADR-001 (wymiary tensora, brak NaN, normalizacja Z-score).
 */
public class AudioPreprocessorTest {

    private AudioPreprocessor preprocessor;
    private StftTransformer stftTransformer;
    private MelFilterBank melFilterBank;

    @Before
    public void setUp() {
        preprocessor = new AudioPreprocessor();
        stftTransformer = new StftTransformer();
        melFilterBank = new MelFilterBank();
    }

    @Test
    public void testFlatTensorOutputDimensions() {
        // Syntetyczna fala sinusoidalna 1000 Hz o długości 66150 próbek (3.0s przy 22050 Hz)
        int numSamples = DspConstants.CHUNK_SAMPLES;
        float[] audio = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            audio[i] = (float) Math.sin(2.0 * Math.PI * 1000.0 * i / DspConstants.SAMPLE_RATE);
        }

        float[] flatTensor = preprocessor.processAudioChunkToFlatTensor(audio);

        assertNotNull("Płaski tensor nie może być nullem", flatTensor);
        assertEquals("Rozmiar tensora musi wynosić dokładnie 99456 elementów (3 * 128 * 259)",
            DspConstants.TENSOR_FLAT_SIZE, flatTensor.length);

        // Weryfikacja braku wartości NaN oraz nieskończoności
        for (int i = 0; i < flatTensor.length; i++) {
            assertFalse("Wykryto NaN na indeksie " + i, Float.isNaN(flatTensor[i]));
            assertFalse("Wykryto nieskończoność na indeksie " + i, Float.isInfinite(flatTensor[i]));
        }
    }

    @Test
    public void testWaveformPeakNormalization() {
        float[] signal = new float[]{2.0f, 4.0f, -6.0f, 8.0f, 10.0f, -2.0f};
        AudioPreprocessor.normalizeWaveformPeak(signal);

        float maxAbs = 0.0f;
        float sum = 0.0f;
        for (float v : signal) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
            sum += v;
        }

        assertEquals("Maksymalna amplituda po normalizacji szczytowej musi wynosić 1.0", 1.0f, maxAbs, 1e-5f);
        assertEquals("Średnia po usunięciu DC offsetu musi wynosić ~0.0", 0.0f, sum / signal.length, 1e-5f);
    }

    @Test
    public void testRmsSilenceThreshold() {
        float[] silence = new float[1000];
        for (int i = 0; i < silence.length; i++) {
            silence[i] = 0.0002f;
        }
        float rmsSilence = AudioPreprocessor.computeRms(silence);
        assertTrue("Cichy sygnał musi być poniżej progu bramki ciszy (0.0004)",
            rmsSilence < DspConstants.RMS_SILENCE_THRESHOLD);

        float[] sound = new float[1000];
        for (int i = 0; i < sound.length; i++) {
            sound[i] = 0.1f;
        }
        float rmsSound = AudioPreprocessor.computeRms(sound);
        assertTrue("Głośny sygnał musi przekraczać próg bramki ciszy (0.0004)",
            rmsSound > DspConstants.RMS_SILENCE_THRESHOLD);
    }

    @Test
    public void testMelFilterBankDimensions() {
        assertEquals("Liczba filtrów Mel musi wynosić 128", 128, melFilterBank.getNumMels());
        assertEquals("Liczba binów FFT musi wynosić 513", 513, melFilterBank.getNumFftBins());
    }

    @Test
    public void testStftDimensionsAndNonNegativePower() {
        float[] signal = new float[DspConstants.CHUNK_SAMPLES];
        for (int i = 0; i < signal.length; i++) {
            signal[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / DspConstants.SAMPLE_RATE);
        }

        float[][] power = stftTransformer.computePowerSpectrogram(signal);

        assertEquals("Liczba binów widma mocy musi wynosić 513", 513, power.length);
        assertEquals("Liczba ramek czasowych dla okna 3.0s musi wynosić 259", 259, power[0].length);

        for (int k = 0; k < 513; k++) {
            for (int t = 0; t < 259; t++) {
                assertTrue("Widmo mocy |X(k)|^2 musi być nieujemne", power[k][t] >= 0.0f);
            }
        }
    }

    @Test
    public void testChannelZScoreNormalization() {
        float[][] channel = new float[128][259];
        for (int r = 0; r < 128; r++) {
            for (int c = 0; c < 259; c++) {
                channel[r][c] = (float) (r * 10 + c);
            }
        }

        AudioPreprocessor.normalizeChannelZScore(channel);

        double sum = 0.0;
        int total = 128 * 259;
        for (float[] r : channel) {
            for (float v : r) {
                sum += v;
            }
        }
        float mean = (float) (sum / total);

        double sumSq = 0.0;
        for (float[] r : channel) {
            for (float v : r) {
                float diff = v - mean;
                sumSq += diff * diff;
            }
        }
        float std = (float) Math.sqrt(sumSq / total);

        assertEquals("Średnia po normalizacji Z-score musi wynosić ~0.0", 0.0f, mean, 1e-4f);
        assertEquals("Odchylenie standardowe po Z-score musi wynosić ~1.0", 1.0f, std, 1e-4f);
    }
}
