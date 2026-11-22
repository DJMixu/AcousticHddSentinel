package com.acoustichddsentinel.dsp;

import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Dedykowany zestaw testów jednostkowych weryfikujący specyficzne przypadki brzegowe DSP (v5):
 * 1. Interpolacja f0: dla f0 = 90.0 Hz daje 90.0 +/- 0.3 Hz (zamiast skwantowanego binu 91.5 Hz).
 * 2. Odrzucanie przydźwięku sieciowego: 60/120 Hz oraz 50/100 Hz nie generują fałszywej rotacji HDD.
 * 3. Detekcja przerywanego pisku stojana (Max-Hold Stall Beep): krótki pisk 2000 Hz (150 ms w oknie 3.0s).
 * 4. Odporność estymatora MAD na gęste, głośne klikanie (Click of Death) przy wyliczaniu transjentów.
 * 5. Uczciwa bramka ciszy RMS < 0.002: brak sztucznego podbijania szumu własnego i zerowanie awarii.
 */
public class DspEdgeCasesTest {

    private RpmHarmonicsAnalyzer rpmAnalyzer;
    private DspPhysicalGuard physicalGuard;
    private AudioPreprocessor preprocessor;

    private static final int SR = DspConstants.SAMPLE_RATE; // 22050 Hz
    private static final int CHUNK_SAMPLES = DspConstants.CHUNK_SAMPLES; // 66150 próbek (3.0s)

    @Before
    public void setUp() {
        rpmAnalyzer = new RpmHarmonicsAnalyzer();
        physicalGuard = new DspPhysicalGuard();
        preprocessor = new AudioPreprocessor();
    }

    /**
     * Test 1: Weryfikacja interpolacji parabolicznej f0.
     * Dla sygnału o częstotliwości 90.0 Hz bez interpolacji bin widma FFT (krok ~5.383 Hz)
     * wskazywałby 17 * 5.3833 = 91.52 Hz (+1.52 Hz błędu).
     * Paraboliczna interpolacja usuwa ten błąd do dokładności 90.0 +/- 0.3 Hz.
     */
    @Test
    public void testF0ParabolicInterpolationAccuracy90Hz() {
        float[] samples = new float[CHUNK_SAMPLES];
        float f0 = 90.0f;

        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            double t = (double) i / SR;
            samples[i] = (float) (
                    0.35 * Math.sin(2.0 * Math.PI * f0 * t) +
                    0.20 * Math.sin(2.0 * Math.PI * 2.0 * f0 * t) +
                    0.15 * Math.sin(2.0 * Math.PI * 3.0 * f0 * t) +
                    0.10 * Math.sin(2.0 * Math.PI * 4.0 * f0 * t) +
                    0.05 * Math.sin(2.0 * Math.PI * 6.0 * f0 * t)
            );
        }

        RpmHarmonicsAnalyzer.RpmResult res = rpmAnalyzer.analyzeChunk(samples);
        assertTrue("Musi poprawnie wykryć rotację wrzeciona 5400 RPM", res.hasRotation);
        assertEquals("5400_rpm", res.detectedRpm);
        assertEquals("Nominalne RPM musi wynosić 5400", 5400, res.nominalRpm);

        // Bez interpolacji: 91.52 Hz. Z interpolacją: 90.0 +/- 0.3 Hz!
        assertEquals("Częstotliwość f0 po interpolacji parabolicznej musi wynosić 90.0 +/- 0.3 Hz",
                90.0f, res.f0FoundHz, 0.3f);
    }

    /**
     * Test 2a: Odrzucanie przydźwięku sieci energetycznej 60 Hz i harmonicznej 120 Hz (USA / Ameryka Pn.).
     * Składowa 120 Hz nie może być mylona z rotacją 7200 RPM silnika wrzeciona.
     */
    @Test
    public void testMainsHumRejection60Hz120Hz() {
        float[] samples = new float[CHUNK_SAMPLES];
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            double t = (double) i / SR;
            // Przydźwięk 60 Hz z wyższymi harmonicznymi sieci (120 Hz, 180 Hz)
            samples[i] = (float) (
                    0.25 * Math.sin(2.0 * Math.PI * 60.0 * t) +
                    0.20 * Math.sin(2.0 * Math.PI * 120.0 * t) +
                    0.08 * Math.sin(2.0 * Math.PI * 180.0 * t) +
                    0.02 * (Math.random() - 0.5)
            );
        }

        RpmHarmonicsAnalyzer.RpmResult res = rpmAnalyzer.analyzeChunk(samples);
        assertFalse("Przydźwięk sieci energetycznej 60/120 Hz NIE może być klasyfikowany jako wirujący HDD", res.hasRotation);
    }

    /**
     * Test 2b: Odrzucanie przydźwięku sieci energetycznej 50 Hz i harmonicznej 100 Hz (Europa).
     * Składowe 50 Hz i 100 Hz nie mogą generować fałszywych detekcji wysokich obrotów.
     */
    @Test
    public void testMainsHumRejection50Hz100Hz() {
        float[] samples = new float[CHUNK_SAMPLES];
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            double t = (double) i / SR;
            // Przydźwięk sieci 50 Hz z harmonicznymi (100 Hz, 150 Hz, 200 Hz, 250 Hz)
            samples[i] = (float) (
                    0.28 * Math.sin(2.0 * Math.PI * 50.0 * t) +
                    0.22 * Math.sin(2.0 * Math.PI * 100.0 * t) +
                    0.10 * Math.sin(2.0 * Math.PI * 150.0 * t) +
                    0.05 * Math.sin(2.0 * Math.PI * 250.0 * t) +
                    0.02 * (Math.random() - 0.5)
            );
        }

        RpmHarmonicsAnalyzer.RpmResult res = rpmAnalyzer.analyzeChunk(samples);
        assertFalse("Przydźwięk sieci energetycznej 50/100 Hz NIE może być klasyfikowany jako wirujący HDD", res.hasRotation);
    }

    /**
     * Test 3: Detekcja przerywanego pisku stojana (Intermittent Stall Beep) za pomocą widma Max-Hold.
     * Symuluje zablokowany silnik wrzeciona, który próbuje ruszyć i emituje pisk 2000 Hz przez zaledwie 150 ms w oknie 3.0s.
     * W klasycznym uśrednianiu Welcha energia ulega rozmyciu (150 ms / 3000 ms = 5%),
     * ale widmo Max-Hold zachowuje pełną amplitudę impulsu, pozwalając na pewną detekcję Stall Beep.
     */
    @Test
    public void testIntermittentStallBeepDetectionViaMaxHold() {
        float[] samples = new float[CHUNK_SAMPLES];
        Random rnd = new Random(12345);

        // Szum tła o niskiej energii
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            samples[i] = (float) (0.005 * (rnd.nextFloat() - 0.5));
        }

        // Krótki impuls pisku stojana 2000 Hz trwający 150 ms (ok. 3307 próbek) w połowie nagrania (od 1.4s do 1.55s)
        int burstStart = (int) (1.4 * SR);
        int burstLen = (int) (0.150 * SR);
        for (int i = 0; i < burstLen; i++) {
            double t = (double) i / SR;
            samples[burstStart + i] += (float) (0.35 * Math.sin(2.0 * Math.PI * 2000.0 * t));
        }

        RpmHarmonicsAnalyzer.RpmResult res = rpmAnalyzer.analyzeChunk(samples);
        assertFalse("Zablokowany silnik z piskiem stojana nie wykazuje rotacji", res.hasRotation);
        assertTrue("Widmo Max-Hold musi wykryć przerywany pisk stojana (150 ms)", res.hasStallBeep);
        assertEquals("Częstotliwość pisku stojana musi wynosić ~2000 Hz", 2000.0f, res.stallBeepFreqHz, 30.0f);
    }

    /**
     * Test 4: Odporność estymatora MAD na gęste, głośne klikanie głowic (Click of Death).
     * W teście symulowane jest 8 ostrych transjentów mechanicznych w oknie 3.0s.
     * Estymator MAD oparty na medianie jest odporny na zniekształcenie progu przez same impulsy klikania,
     * poprawnie zliczając transjenty mechaniczne.
     */
    @Test
    public void testMadTransientDetectionRobustnessUnderDenseClicking() {
        float[] samples = new float[CHUNK_SAMPLES];
        Random rnd = new Random(42);

        // Szum bazowy
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            samples[i] = (float) (0.01 * (rnd.nextFloat() - 0.5));
        }

        // 8 impulsów uderzeniowych głowicy co ~350 ms
        int numClicks = 8;
        for (int c = 0; c < numClicks; c++) {
            int clickCenter = (int) ((0.25 + c * 0.35) * SR);
            // Ostry impuls akustyczny o szerokości ~5 ms
            for (int k = -50; k <= 50; k++) {
                int idx = clickCenter + k;
                if (idx >= 0 && idx < CHUNK_SAMPLES) {
                    float window = (float) Math.exp(-0.5 * (k * k) / (15.0 * 15.0));
                    samples[idx] += 0.85f * window * ((k % 2 == 0) ? 1.0f : -1.0f);
                }
            }
        }

        // Zliczanie transjentów z estymatorem MAD
        int detectedTransients = DspPhysicalGuard.countTransientPeaks(samples, SR);
        assertTrue("Estymator MAD musi wykryć co najmniej 7 z 8 transjentów klikania (wykryto: " + detectedTransients + ")",
                detectedTransients >= 7);

        // Weryfikacja sanityzacji: gęste klikanie o wysokim CF (>= 3.5) NIE może zostać zawetowane
        float[] rawProbs = new float[]{0.05f, 0.80f, 0.10f, 0.05f}; // clicking = 0.80
        DspPhysicalGuard.SanitizationResult res = physicalGuard.sanitizeProbabilities(
                rawProbs, samples, null, null
        );

        assertFalse("Autentyczne gęste klikanie głowic NIE może otrzymać weta clicking", res.vetoClicking);
        assertTrue("Prawdopodobieństwo clicking musi pozostać wysokie (>= 0.75)", res.sanitizedProbs[1] >= 0.75f);
    }

    /**
     * Test 5: Uczciwa bramka ciszy RMS < 0.002.
     * Wycinek o RMS < 0.002 (np. 0.001f) reprezentuje szum własny toru audio lub brak kontaktu z dyskiem.
     * Sprawdza:
     * 1. RpmHarmonicsAnalyzer zwraca hasRotation = false i hasStallBeep = false.
     * 2. DspPhysicalGuard wymusza normal = 1.0 oraz zeruje prawdopodobieństwa awarii.
     * 3. AudioPreprocessor.processAudioChunkToFlatTensor z outFlatTensor działa Zero-GC bez wyjątków.
     */
    @Test
    public void testSilenceGateHonestyRmsBelowThreshold() {
        float[] quietSamples = new float[CHUNK_SAMPLES];
        // Sygnał o RMS = 0.001f (poniżej progu ciszy 0.002f)
        for (int i = 0; i < CHUNK_SAMPLES; i++) {
            quietSamples[i] = (i % 2 == 0) ? 0.001f : -0.001f;
        }

        float rms = AudioPreprocessor.computeRms(quietSamples);
        assertTrue("RMS sygnału musi być poniżej progu 0.002", rms < DspConstants.RMS_SILENCE_THRESHOLD);

        // 1. Sprawdzenie analizatora RPM
        RpmHarmonicsAnalyzer.RpmResult rpmRes = rpmAnalyzer.analyzeChunk(quietSamples);
        assertFalse("Cisza nie może zgłaszać rotacji silnika", rpmRes.hasRotation);
        assertFalse("Cisza nie może zgłaszać pisku stojana", rpmRes.hasStallBeep);

        // 2. Sprawdzenie Strażnika Fizycznego
        float[] rawFaultProbs = new float[]{0.30f, 0.30f, 0.10f, 0.30f};
        DspPhysicalGuard.SanitizationResult guardRes = physicalGuard.sanitizeProbabilities(
                rawFaultProbs, quietSamples, null, rpmRes
        );

        assertEquals("Prawdopodobieństwo normal musi wynosić 1.0 przy ciszy", 1.0f, guardRes.sanitizedProbs[2], 1e-5f);
        assertEquals("Prawdopodobieństwo beeping musi wynosić 0.0 przy ciszy", 0.0f, guardRes.sanitizedProbs[0], 1e-5f);
        assertEquals("Prawdopodobieństwo clicking musi wynosić 0.0 przy ciszy", 0.0f, guardRes.sanitizedProbs[1], 1e-5f);
        assertEquals("Prawdopodobieństwo scratching musi wynosić 0.0 przy ciszy", 0.0f, guardRes.sanitizedProbs[3], 1e-5f);

        // 3. Sprawdzenie dwuargumentowej metody Zero-GC preprocesora
        float[] outTensor = new float[DspConstants.TENSOR_FLAT_SIZE];
        preprocessor.processAudioChunkToFlatTensor(quietSamples, outTensor);
        assertNotNull(outTensor);
        assertEquals(DspConstants.TENSOR_FLAT_SIZE, outTensor.length);
        assertFalse("Tensor nie może zawierać NaN", Float.isNaN(outTensor[0]));
    }
}
