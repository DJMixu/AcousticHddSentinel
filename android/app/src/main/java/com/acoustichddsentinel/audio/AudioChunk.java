package com.acoustichddsentinel.audio;

/**
 * Kontener pojedynczego wycinka audio zarejestrowanego przez mikrofon sprzętowy.
 */
public final class AudioChunk {

    private final int windowIndex;
    private final float[] samples;
    private final int sampleRate;
    private final float rms;
    private final long timestamp;

    public AudioChunk(int windowIndex, float[] samples, int sampleRate, float rms, long timestamp) {
        this.windowIndex = windowIndex;
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.rms = rms;
        this.timestamp = timestamp;
    }

    public int getWindowIndex() {
        return windowIndex;
    }

    public float[] getSamples() {
        return samples;
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public float getRms() {
        return rms;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public float getDurationSeconds() {
        return (float) samples.length / sampleRate;
    }
}
