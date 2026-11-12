package com.acoustichddsentinel.dsp;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test jednostkowy dla RpmHarmonicsAnalyzer w czystej Javie.
 * Weryfikuje:
 * 1. Detekcję 5400 RPM (90 Hz fundamentalna + harmoniczne)
 * 2. Detekcję 7200 RPM (120 Hz fundamentalna + harmoniczne)
 * 3. Odrzucanie szumu wentylatora PC (140 Hz BPF + szum)
 * 4. Wykrywanie zatarcia silnika (Stall Beep 1500 Hz)
 * 5. Zachowanie przy ciszy (RMS < 0.002)
 */
public class RpmHarmonicsAnalyzerTest {

    private RpmHarmonicsAnalyzer analyzer;
    private int sampleRate;
    private int numSamples;

    @Before
    public void setUp() {
        analyzer = new RpmHarmonicsAnalyzer();
        sampleRate = DspConstants.SAMPLE_RATE; // 22050
        numSamples = DspConstants.CHUNK_SAMPLES; // 66150 (3.0s)
    }

    @Test
    public void test5400RpmDetection() {
        float[] samples = new float[numSamples];
        float f0 = 90.0f;

        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / sampleRate;
            double signal = 0.35 * Math.sin(2.0 * Math.PI * f0 * t)
                          + 0.22 * Math.sin(2.0 * Math.PI * 2.0 * f0 * t)
                          + 0.15 * Math.sin(2.0 * Math.PI * 3.0 * f0 * t)
                          + 0.10 * Math.sin(2.0 * Math.PI * 4.0 * f0 * t)
                          + 0.05 * Math.sin(2.0 * Math.PI * 6.0 * f0 * t)
                          + 0.02 * (Math.random() - 0.5);
            samples[i] = (float) signal;
        }

        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(samples);
        assertTrue("Musi wykryć rotację silnika 5400 RPM", res.hasRotation);
        assertEquals("5400_rpm", res.detectedRpm);
        assertEquals(5400, res.nominalRpm);
        assertFalse("Nie może zgłosić zablokowania przy prawidłowej rotacji", res.hasStallBeep);
        assertTrue(res.rpmConfidence > 0.4f);
    }

    @Test
    public void test7200RpmDetection() {
        float[] samples = new float[numSamples];
        float f0 = 120.0f;

        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / sampleRate;
            double signal = 0.38 * Math.sin(2.0 * Math.PI * f0 * t)
                          + 0.25 * Math.sin(2.0 * Math.PI * 2.0 * f0 * t)
                          + 0.16 * Math.sin(2.0 * Math.PI * 3.0 * f0 * t)
                          + 0.10 * Math.sin(2.0 * Math.PI * 4.0 * f0 * t)
                          + 0.06 * Math.sin(2.0 * Math.PI * 6.0 * f0 * t)
                          + 0.02 * (Math.random() - 0.5);
            samples[i] = (float) signal;
        }

        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(samples);
        assertTrue("Musi wykryć rotację silnika 7200 RPM", res.hasRotation);
        assertEquals("7200_rpm", res.detectedRpm);
        assertEquals(7200, res.nominalRpm);
        assertFalse(res.hasStallBeep);
    }

    @Test
    public void testPcFanRejection() {
        float[] samples = new float[numSamples];
        float bpf = 145.0f; // BPF wentylatora PC (nie 90 ani 120 Hz!)

        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / sampleRate;
            double signal = 0.08 * Math.sin(2.0 * Math.PI * bpf * t)
                          + 0.25 * (Math.random() - 0.5); // dominujący szum turbulentny
            samples[i] = (float) signal;
        }

        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(samples);
        assertFalse("Wentylator PC nie może być uznany za wirujący dysk HDD", res.hasRotation);
        assertNull(res.detectedRpm);
    }

    @Test
    public void testSeizedSpindleStallBeep() {
        float[] samples = new float[numSamples];
        float beepFreq = 1500.0f; // pisk stojana zablokowanego silnika

        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / sampleRate;
            double signal = 0.30 * Math.sin(2.0 * Math.PI * beepFreq * t)
                          + 0.08 * Math.sin(2.0 * Math.PI * 3000.0 * t)
                          + 0.01 * (Math.random() - 0.5);
            samples[i] = (float) signal;
        }

        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(samples);
        assertFalse(res.hasRotation);
        assertTrue("Musi wykryć pisk stojana zatartego silnika", res.hasStallBeep);
        assertEquals(1500.0f, res.stallBeepFreqHz, 25.0f);
    }

    @Test
    public void testSilenceRejection() {
        float[] silentSamples = new float[numSamples]; // same zera
        RpmHarmonicsAnalyzer.RpmResult res = analyzer.analyzeChunk(silentSamples);
        assertFalse(res.hasRotation);
        assertFalse(res.hasStallBeep);
        assertNull(res.detectedRpm);
    }
}
