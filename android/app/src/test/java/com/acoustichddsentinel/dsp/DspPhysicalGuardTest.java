package com.acoustichddsentinel.dsp;

import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Zestaw testów jednostkowych dla klasy DspPhysicalGuard w czystej Javie 17 (JUnit 4).
 * Weryfikuje twarde reguły weta fizycznego (ADR-001):
 * 1. Odrzucanie fałszywego klikania (clicking) przy jednostajnym szumie wentylatora PC / oczyszczacza.
 * 2. Odrzucanie fałszywego tarcia (scratching) przy przesterowaniu mikrofonu (hard clipping).
 * 3. Działanie Smart Clipping Bypass przy autentycznym zderzeniu głowic (Crash Stop).
 * 4. Ochrona pisku stojana (stall beep) zatartego silnika przy braku obrotów.
 * 5. Weto dla pisku stojana (beeping), gdy wykryto potwierdzoną rotację wrzeciona (5400/7200 RPM).
 * 6. Weto i zerowanie prawdopodobieństwa awarii dla cichych ramek (RMS < 0.002).
 */
public class DspPhysicalGuardTest {

    private DspPhysicalGuard guard;
    private static final int SR = DspConstants.SAMPLE_RATE; // 22050 Hz
    private static final int N_SAMPLES = DspConstants.CHUNK_SAMPLES; // 66150 (3.0s)

    @Before
    public void setUp() {
        guard = new DspPhysicalGuard();
    }

    @Test
    public void testFanNoiseVetoesClicking() {
        // Syntetyczny szum wentylatora PC: niskie pasmo (150-400 Hz), niski Crest Factor, brak transjentów
        float[] samples = new float[N_SAMPLES];
        Random rnd = new Random(123);
        for (int i = 0; i < N_SAMPLES; i++) {
            float t = (float) i / SR;
            // Dominuje łagodny ton łopatek 180 Hz oraz szum o niskiej dynamice
            samples[i] = (float) (0.15 * Math.sin(2.0 * Math.PI * 180.0 * t)
                    + 0.08 * Math.sin(2.0 * Math.PI * 360.0 * t)
                    + 0.03 * (rnd.nextFloat() - 0.5f));
        }

        float[] rawProbs = new float[]{0.05f, 0.85f, 0.05f, 0.05f}; // fałszywy alarm CNN: clicking = 0.85
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, samples, null, null);

        assertNotNull("Wynik sanityzacji nie może być null", res);
        assertTrue("Sygnał musi zostać zakwalifikowany jako wentylator/oczyszczacz", res.isAerodynamicFanOrPurifier);
        assertTrue("Weto CLICKING musi być nałożone na szum wentylatora", res.vetoClicking);
        assertTrue("Prawdopodobieństwo clicking musi zostać zredukowane <= 0.02", res.sanitizedProbs[1] <= 0.02f);
        assertTrue("Prawdopodobieństwo klasy normal musi zostać podniesione >= 0.80", res.sanitizedProbs[2] >= 0.80f);
    }

    @Test
    public void testMicrophoneClippingVetoesScratching() {
        // Przesterowany sygnał basowy (np. dmuchnięcie w mikrofon lub uderzenie w obudowę telefonu)
        // Cechy: peak >= 0.35, > 2% próbek obciętych na granicy nasycenia, ale brak wysokich częstotliwości (>4 kHz)
        float[] samples = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            float t = (float) i / SR;
            float rawVal = (float) (1.8 * Math.sin(2.0 * Math.PI * 150.0 * t)); // silna amplituda powodująca obcięcie
            if (rawVal > 0.98f) samples[i] = 0.98f;
            else if (rawVal < -0.98f) samples[i] = -0.98f;
            else samples[i] = rawVal;
        }

        float[] rawProbs = new float[]{0.05f, 0.05f, 0.05f, 0.85f}; // fałszywy alarm CNN: scratching = 0.85
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, samples, null, null);

        assertNotNull(res);
        assertTrue("Przesterowanie mikrofonu musi zostać wykryte", res.isMicrophoneClipped);
        assertFalse("Smart Clipping Bypass NIE może być aktywny przy braku energii >4 kHz", res.smartClippingBypass);
        assertTrue("Weto SCRATCHING musi zostać nałożone na przesterowanie toru audio", res.vetoScratching);
        assertTrue("Prawdopodobieństwo scratching musi spaść <= 0.02", res.sanitizedProbs[3] <= 0.02f);
        assertTrue("Prawdopodobieństwo normal musi zostać zrekompensowane", res.sanitizedProbs[2] >= 0.80f);
    }

    @Test
    public void testSmartClippingBypassPreservesScratchingOnHighCrestAndHighFreq() {
        // Katastrofalne tarcie głowic o wirujący talerz (Crash Stop / Severe Head Crash):
        // 1. Ekstremalny Crest Factor (>= 4.5)
        // 2. Bardzo wysoki udział energii w paśmie tarcia >4 kHz (> 18%)
        // 3. Występowanie clippingu przetwornika ADC
        float[] samples = new float[N_SAMPLES];
        Random rnd = new Random(42);
        for (int i = 0; i < N_SAMPLES; i++) {
            float t = (float) i / SR;
            // Silne tarcie wysokoczęstotliwościowe 4.8 kHz i 7.2 kHz
            samples[i] = (float) (0.12 * (Math.sin(2.0 * Math.PI * 4800.0 * t) + Math.sin(2.0 * Math.PI * 7200.0 * t))
                    + 0.05 * (rnd.nextFloat() - 0.5f));
        }

        // Dodanie ostrych impulsów uderzeniowych z obcięciem nasycenia (> 2% próbek)
        for (int p = 0; p < 30; p++) {
            int pos = 1000 + p * 2000;
            for (int k = 0; k < 50; k++) {
                if (pos + k < N_SAMPLES) {
                    samples[pos + k] = (k % 2 == 0) ? 0.99f : -0.99f;
                }
            }
        }

        float[] rawProbs = new float[]{0.05f, 0.05f, 0.10f, 0.80f}; // scratching = 0.80
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, samples, null, null);

        assertNotNull(res);
        assertTrue("Sygnał uderzeniowy musi wykazać przesterowanie", res.isMicrophoneClipped);
        assertTrue("Smart Clipping Bypass MUSI zadziałać dla ostrego tarcia (CF >= 4.5 & HighFreq > 18%)", res.smartClippingBypass);
        assertFalse("Weto scratching NIE może być nałożone przy prawdziwym Crash Stopie", res.vetoScratching);
        assertTrue("Prawdopodobieństwo awarii scratching musi pozostać wysokie", res.sanitizedProbs[3] >= 0.70f);
    }

    @Test
    public void testSeizedSpindleStallBeepProtected() {
        // Pisk zatartego stojana silnika wrzeciona (1500 Hz): brak rotacji, obecny pisk stojana > 1050 Hz
        float[] samples = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            float t = (float) i / SR;
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 1500.0 * t)
                    + 0.08 * Math.sin(2.0 * Math.PI * 3000.0 * t));
        }

        RpmHarmonicsAnalyzer.RpmResult rpmRes = new RpmHarmonicsAnalyzer.RpmResult(
                false,        // brak rotacji wrzeciona
                null,
                0,
                0.0f,
                0.0f,
                true,         // obecny pisk stojana
                1500.0f,      // częstotliwość pisku stojana
                0.15f
        );

        float[] rawProbs = new float[]{0.85f, 0.05f, 0.05f, 0.05f}; // beeping = 0.85
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, samples, null, rpmRes);

        assertNotNull(res);
        assertFalse("Pisk stojana (beeping) bez rotacji NIE MOŻE być zawetowany", res.vetoBeeping);
        assertTrue("Prawdopodobieństwo beeping musi pozostać wysokie (>= 0.80)", res.sanitizedProbs[0] >= 0.80f);
    }

    @Test
    public void testNormalSpindleRotationVetoesBeeping() {
        // Prawidłowo wirujący dysk 7200 RPM: rotacja wrzeciona potwierdzona przez analizator harmonicznych
        float[] samples = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            float t = (float) i / SR;
            samples[i] = (float) (0.20 * Math.sin(2.0 * Math.PI * 120.0 * t)
                    + 0.12 * Math.sin(2.0 * Math.PI * 240.0 * t)
                    + 0.08 * Math.sin(2.0 * Math.PI * 360.0 * t));
        }

        RpmHarmonicsAnalyzer.RpmResult rpmRes = new RpmHarmonicsAnalyzer.RpmResult(
                true,         // potwierdzona prawidłowa rotacja silnika
                "7200_rpm",
                7200,
                0.90f,
                120.0f,
                false,        // brak pisku stojana
                0.0f,
                0.15f
        );

        // Fałszywy alarm CNN na klasę beeping (np. z powodu piszczącej cewki VRM w tle)
        float[] rawProbs = new float[]{0.80f, 0.05f, 0.10f, 0.05f};
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, samples, null, rpmRes);

        assertNotNull(res);
        assertTrue("Weto BEEPING musi zostać nałożone, gdy silnik normalnie wiruje", res.vetoBeeping);
        assertTrue("Prawdopodobieństwo beeping musi spaść <= 0.02", res.sanitizedProbs[0] <= 0.02f);
        assertTrue("Prawdopodobieństwo klasy normal musi wzrosnąć >= 0.80", res.sanitizedProbs[2] >= 0.80f);
    }

    @Test
    public void testLowRmsSilenceRejection() {
        // Cichy sygnał o energii RMS poniżej progu ciszy (RMS < 0.002)
        float[] silentSamples = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            silentSamples[i] = 0.0003f; // RMS ≈ 0.0003 < 0.002
        }

        // Wejściowe prawdopodobieństwa symulujące fałszywy alarm w ciszy
        float[] rawProbs = new float[]{0.30f, 0.30f, 0.10f, 0.30f};
        DspPhysicalGuard.SanitizationResult res = guard.sanitizeProbabilities(rawProbs, silentSamples, null, null);

        assertNotNull(res);
        float faultProb = res.sanitizedProbs[0] + res.sanitizedProbs[1] + res.sanitizedProbs[3];
        assertEquals("Sygnał w ciszy musi zachować dokładnie 0.0 prawdopodobieństwa awarii", 0.0f, faultProb, 1e-6f);
        assertEquals("Sygnał w ciszy musi mieć 100% prawdopodobieństwa klasy normal", 1.0f, res.sanitizedProbs[2], 1e-6f);
    }
}
