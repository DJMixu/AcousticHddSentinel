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
    public void testQuietAudioBoosterRange() {
        // 1. Bardzo cichy sygnał (RMS = 0.0008, poniżej starej bramki 0.002, ale powyżej 0.0003)
        float[] quiet = new float[1000];
        java.util.Arrays.fill(quiet, 0.0008f);
        float[] boosted = AudioPreprocessor.checkAndBoostAudio(quiet, DspConstants.TARGET_BOOST_RMS);
        float boostedRms = AudioPreprocessor.computeRms(boosted);
        assertTrue("Cichy sygnał (0.0008) musi zostać podbity powyżej progu ciszy 0.002",
            boostedRms >= DspConstants.RMS_SILENCE_THRESHOLD);
        assertEquals("Podbicie powinno osiągnąć cel TARGET_BOOST_RMS",
            DspConstants.TARGET_BOOST_RMS, boostedRms, 1e-4f);

        // 2. Szum cyfrowy / cisza bezwzględna (RMS = 0.0001 < 0.0003)
        float[] silence = new float[1000];
        java.util.Arrays.fill(silence, 0.0001f);
        float[] unboosted = AudioPreprocessor.checkAndBoostAudio(silence, DspConstants.TARGET_BOOST_RMS);
        float unboostedRms = AudioPreprocessor.computeRms(unboosted);
        assertEquals("Cisza cyfrowa < 0.0003 nie powinna być sztucznie podbijana",
            0.0001f, unboostedRms, 1e-5f);
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

    @Test
    public void testDynamicHarmonicTrackingDrifted5400() {
        RpmHarmonicsAnalyzer analyzer = new RpmHarmonicsAnalyzer();
        int nSamples = DspConstants.CHUNK_SAMPLES;
        float[] samples = new float[nSamples];

        // Symulacja agonalnego dysku 2.5": silnik zwolnił z 90 Hz do 87.5 Hz (-2.5 Hz dryfu)
        float f0 = 87.5f;
        for (int i = 0; i < nSamples; i++) {
            float t = (float) i / DspConstants.SAMPLE_RATE;
            samples[i] = (float) (
                0.25 * Math.sin(2.0 * Math.PI * f0 * t) +
                0.15 * Math.sin(2.0 * Math.PI * (2.0 * f0) * t) +
                0.12 * Math.sin(2.0 * Math.PI * (3.0 * f0) * t) +
                0.10 * Math.sin(2.0 * Math.PI * (4.0 * f0) * t) +
                0.08 * Math.sin(2.0 * Math.PI * (6.0 * f0) * t)
            );
        }

        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(samples);
        assertTrue("Dynamic Harmonic Tracking musi wykryć rotację dla silnika z dryfem 87.5 Hz", res.hasRotation);
        assertEquals("Wykryty profil musi odpowiadać 5400 RPM", "5400_rpm", res.detectedRpm);
        assertEquals("Nominalne RPM musi wynosić 5400", 5400, res.nominalRpm);
        assertTrue("Wykryte f0 powinno być bliskie 87.5 Hz (+/- 3 Hz)", Math.abs(res.f0FoundHz - 87.5f) < 3.0f);
    }

    @Test
    public void testSmartClippingBypassOnCrashStopImpact() {
        DspPhysicalGuard guard = new DspPhysicalGuard();
        int nSamples = DspConstants.CHUNK_SAMPLES;
        float[] samples = new float[nSamples];

        // Syntetyczny sygnał katastrofalnego uderzenia głowicy (Crash Stop):
        // 1. Wysoka dynamika impulsowa (Crest Factor >= 4.5)
        // 2. Szerokopasmowa energia tarcia w paśmie > 4 kHz (highFreqFraction > 0.18)
        // 3. Obcięcie wierzchołków sygnału przez przetwornik ADC (clipping)
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < nSamples; i++) {
            float t = (float) i / DspConstants.SAMPLE_RATE;
            samples[i] = (float) (0.12 * (Math.sin(2.0 * Math.PI * 4800.0 * t) + Math.sin(2.0 * Math.PI * 7200.0 * t))
                    + 0.05 * (rnd.nextFloat() - 0.5));
        }

        // Dodanie ostrych impulsów uderzeniowych z przesterowaniem (> 2% próbek osiąga szczyt)
        for (int p = 0; p < 30; p++) {
            int pos = 1000 + p * 2000;
            for (int k = 0; k < 50; k++) {
                if (pos + k < nSamples) {
                    samples[pos + k] = (k % 2 == 0) ? 0.99f : -0.99f; // Przesterowanie szczytowe
                }
            }
        }

        float[] rawProbs = new float[]{0.05f, 0.75f, 0.10f, 0.10f}; // clicking = 0.75
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(
            rawProbs, samples, null, null
        );

        assertTrue("Sygnał musi zostać zidentyfikowany jako przesterowany", res.isMicrophoneClipped);
        assertTrue("Smart Clipping Bypass musi być aktywny przy CF >= 4.5 i highFreq > 18%", res.smartClippingBypass);
        assertFalse("Weto clicking NIE może zostać nałożone przy autentycznym Crash Stopie", res.vetoClicking);
        assertTrue("Prawdopodobieństwo clicking musi pozostać wysokie", res.sanitizedProbs[1] >= 0.70f);
    }
}
