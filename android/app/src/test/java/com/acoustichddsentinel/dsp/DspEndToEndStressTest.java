package com.acoustichddsentinel.dsp;

import org.junit.Before;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy integracyjne i obciążeniowe potoku DSP dla różnych typów sygnałów.
 */
public class DspEndToEndStressTest {

    private AudioPreprocessor preprocessor;

    @Before
    public void setUp() {
        preprocessor = new AudioPreprocessor();
    }

    @Test
    public void testChirpSweepSignal() {
        int n = DspConstants.CHUNK_SAMPLES;
        float[] chirp = new float[n];
        // Liniowy sweep częstotliwości od 100 Hz do 8000 Hz w czasie 3.0 s
        for (int i = 0; i < n; i++) {
            double t = (double) i / DspConstants.SAMPLE_RATE;
            double freq = 100.0 + (8000.0 - 100.0) * (t / DspConstants.CHUNK_DURATION_SEC);
            chirp[i] = (float) Math.sin(2.0 * Math.PI * freq * t);
        }

        float[] tensor = preprocessor.processAudioChunkToFlatTensor(chirp);
        assertEquals(DspConstants.TENSOR_FLAT_SIZE, tensor.length);

        for (float v : tensor) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }

        float[][] viz = preprocessor.extractLogMelForVisualization(chirp);
        assertEquals(128, viz.length);
        assertEquals(259, viz[0].length);
    }

    @Test
    public void testImpulsiveClickSignal() {
        int n = DspConstants.CHUNK_SAMPLES;
        float[] clicks = new float[n];
        // Impulsy co 0.5 sekundy (symulacja uderzeń głowic)
        int step = DspConstants.SAMPLE_RATE / 2;
        for (int i = 0; i < n; i += step) {
            clicks[i] = 1.0f;
            if (i + 1 < n) clicks[i + 1] = -0.8f;
            if (i + 2 < n) clicks[i + 2] = 0.5f;
        }

        float[] tensor = preprocessor.processAudioChunkToFlatTensor(clicks);
        assertEquals(DspConstants.TENSOR_FLAT_SIZE, tensor.length);

        for (float v : tensor) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }
    }

    @Test
    public void testRandomNoiseSignal() {
        int n = DspConstants.CHUNK_SAMPLES;
        float[] noise = new float[n];
        Random rnd = new Random(42);
        for (int i = 0; i < n; i++) {
            noise[i] = (rnd.nextFloat() * 2.0f) - 1.0f;
        }

        float[] tensor = preprocessor.processAudioChunkToFlatTensor(noise);
        assertEquals(DspConstants.TENSOR_FLAT_SIZE, tensor.length);

        for (float v : tensor) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }
    }
}
