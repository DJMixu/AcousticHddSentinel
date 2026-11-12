package com.acoustichddsentinel.audio;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Testy jednostkowe obiektu kontenera AudioChunk.
 */
public class AudioChunkTest {

    @Test
    public void testAudioChunkProperties() {
        float[] samples = new float[66150]; // 3.0s przy 22050 Hz
        long now = System.currentTimeMillis();

        AudioChunk chunk = new AudioChunk(1, samples, 22050, 0.045f, now);

        assertEquals(1, chunk.getWindowIndex());
        assertNotNull(chunk.getSamples());
        assertEquals(66150, chunk.getSamples().length);
        assertEquals(22050, chunk.getSampleRate());
        assertEquals(0.045f, chunk.getRms(), 1e-5f);
        assertEquals(now, chunk.getTimestamp());
        assertEquals(3.0f, chunk.getDurationSeconds(), 1e-4f);
    }
}
