package com.acoustichddsentinel.audio;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import com.acoustichddsentinel.dsp.AudioPreprocessor;
import com.acoustichddsentinel.dsp.DspConstants;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Rejestrator sygnału audio z mikrofonu sprzętowego smartfona (android.media.AudioRecord).
 * Obsługuje rejestrację od 5 do 30 sekund, próbkowanie 22050 Hz (z automatycznym fallbackiem do 44100 Hz),
 * przeliczanie RMS w czasie rzeczywistym oraz podział na 3-sekundowe wycinki analityczne.
 */
public class AudioRecorder {

    private static final String TAG = "AudioRecorder";

    public interface RecordCallback {
        void onProgress(int elapsedMs, int totalMs, float currentRms);
        void onRecordingComplete(List<AudioChunk> chunks, float overallRms);
        void onError(String message);
    }

    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private Thread recordingThread;
    private AudioRecord audioRecord;

    public boolean isRecording() {
        return isRecording.get();
    }

    /**
     * Rozpoczyna rejestrację audio w osobnym wątku roboczym.
     *
     * @param durationSeconds żądany czas rejestracji (5 do 30 sekund)
     * @param callback        odbiornik zdarzeń postępu i wyniku
     */
    @SuppressLint("MissingPermission")
    public synchronized void startRecording(int durationSeconds, RecordCallback callback) {
        if (isRecording.get()) {
            callback.onError("Rejestracja jest już w toku.");
            return;
        }

        int targetSr = DspConstants.SAMPLE_RATE; // 22050 Hz
        int hardwareSr = targetSr;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;

        int minBufferSize = AudioRecord.getMinBufferSize(hardwareSr, channelConfig, audioFormat);
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            // Sprzęt może nie wspierać natywnie 22050 Hz - fallback do 44100 Hz
            hardwareSr = DspConstants.FALLBACK_SAMPLE_RATE;
            minBufferSize = AudioRecord.getMinBufferSize(hardwareSr, channelConfig, audioFormat);
        }

        if (minBufferSize <= 0) {
            callback.onError("Sprzęt audio nie obsługuje wymaganych parametrów rejestracji.");
            return;
        }

        int bufferSize = Math.max(minBufferSize * 2, 4096);
        try {
            audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                hardwareSr,
                channelConfig,
                audioFormat,
                bufferSize
            );
        } catch (IllegalArgumentException e) {
            callback.onError("Błąd inicjalizacji AudioRecord: " + e.getMessage());
            return;
        }

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            callback.onError("Nie udało się zainicjalizować mikrofonu sprzętowego.");
            audioRecord.release();
            audioRecord = null;
            return;
        }

        final int finalHardwareSr = hardwareSr;
        isRecording.set(true);

        recordingThread = new Thread(() -> {
            try {
                audioRecord.startRecording();
            } catch (IllegalStateException e) {
                isRecording.set(false);
                callback.onError("Błąd startu nagrywania: " + e.getMessage());
                return;
            }

            int totalTargetSamples = durationSeconds * DspConstants.SAMPLE_RATE;
            int totalExpectedHardwareSamples = durationSeconds * finalHardwareSr;

            List<Float> collectedHardwareSamples = new ArrayList<>(totalExpectedHardwareSamples);
            short[] readBuffer = new short[bufferSize / 2];
            long startTime = System.currentTimeMillis();
            long totalDurationMs = durationSeconds * 1000L;

            while (isRecording.get() && (System.currentTimeMillis() - startTime) < totalDurationMs) {
                int readShorts = audioRecord.read(readBuffer, 0, readBuffer.length);
                if (readShorts > 0) {
                    float sumSq = 0.0f;
                    for (int i = 0; i < readShorts; i++) {
                        // Konwersja 16-bit PCM [-32768, 32767] do float [-1.0, 1.0]
                        float sample = readBuffer[i] / 32768.0f;
                        collectedHardwareSamples.add(sample);
                        sumSq += sample * sample;
                    }

                    float currentRms = (float) Math.sqrt(sumSq / readShorts);
                    int elapsedMs = (int) (System.currentTimeMillis() - startTime);
                    callback.onProgress(elapsedMs, (int) totalDurationMs, currentRms);
                } else if (readShorts < 0) {
                    Log.w(TAG, "AudioRecord błąd odczytu próbek: " + readShorts);
                    break;
                }
            }

            // Zatrzymanie sprzętu
            stopInternal();

            if (collectedHardwareSamples.isEmpty()) {
                callback.onError("Nie zarejestrowano żadnych próbek audio.");
                return;
            }

            // Konwersja z List<Float> do float[]
            float[] recordedSamples = new float[collectedHardwareSamples.size()];
            for (int i = 0; i < recordedSamples.length; i++) {
                recordedSamples[i] = collectedHardwareSamples.get(i);
            }

            // Decymacja 2:1 z 44100 Hz do 22050 Hz jeśli nagrywano na 44100 Hz
            float[] final22kSamples;
            if (finalHardwareSr == DspConstants.FALLBACK_SAMPLE_RATE) {
                final22kSamples = resample44kTo22k(recordedSamples);
            } else {
                final22kSamples = recordedSamples;
            }

            float overallRms = AudioPreprocessor.computeRms(final22kSamples);

            // Weryfikacja bramki ciszy (RMS Gate >= 0.002)
            if (overallRms < DspConstants.RMS_SILENCE_THRESHOLD) {
                Log.w(TAG, "Ostrzeżenie: Poziom RMS " + overallRms + " < " + DspConstants.RMS_SILENCE_THRESHOLD);
            }

            // Podział nagrania na 3-sekundowe okna analityczne (66150 próbek każde)
            List<AudioChunk> chunks = sliceInto3sChunks(final22kSamples);
            callback.onRecordingComplete(chunks, overallRms);

        }, "AudioRecorderThread");

        recordingThread.start();
    }

    /**
     * Zatrzymuje bieżącą rejestrację audio.
     */
    public synchronized void stopRecording() {
        if (!isRecording.get()) {
            return;
        }
        isRecording.set(false);
        stopInternal();
    }

    private void stopInternal() {
        if (audioRecord != null) {
            try {
                if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop();
                }
                audioRecord.release();
            } catch (Exception e) {
                Log.e(TAG, "Błąd podczas zwalniania AudioRecord", e);
            } finally {
                audioRecord = null;
            }
        }
    }

    /**
     * Decymacja antyaliasingowa 2:1 z 44.1 kHz do 22.05 kHz.
     */
    private static float[] resample44kTo22k(float[] input44k) {
        int outLen = input44k.length / 2;
        float[] out22k = new float[outLen];
        for (int i = 0; i < outLen; i++) {
            // Prosty 2-punktowy filtr uśredniający (Box filter / Anti-Aliasing)
            out22k[i] = 0.5f * (input44k[2 * i] + input44k[2 * i + 1]);
        }
        return out22k;
    }

    /**
     * Dzieli sygnał o długości 5-30s na 3-sekundowe wycinki (66150 próbek).
     * Jeśli końcówka jest krótsza niż 3s, jest dopełniana zerami lub pomijana.
     */
    private static List<AudioChunk> sliceInto3sChunks(float[] samples) {
        List<AudioChunk> chunks = new ArrayList<>();
        int chunkSize = DspConstants.CHUNK_SAMPLES; // 66150 próbek
        int step = chunkSize; // Czyste okna bez nakładania w ewaluacji

        int totalLen = samples.length;
        int windowIndex = 0;

        for (int start = 0; start + chunkSize <= totalLen; start += step) {
            float[] chunkSamples = new float[chunkSize];
            System.arraycopy(samples, start, chunkSamples, 0, chunkSize);
            float chunkRms = AudioPreprocessor.computeRms(chunkSamples);
            chunks.add(new AudioChunk(windowIndex++, chunkSamples, DspConstants.SAMPLE_RATE, chunkRms, System.currentTimeMillis()));
        }

        // Jeżeli sygnał miał np. 5 sekund, pierwsze okno to [0..3s], drugie okno to [2..5s] lub z resztą
        if (chunks.isEmpty() && totalLen > 0) {
            float[] chunkSamples = new float[chunkSize];
            System.arraycopy(samples, 0, chunkSamples, 0, Math.min(totalLen, chunkSize));
            float chunkRms = AudioPreprocessor.computeRms(chunkSamples);
            chunks.add(new AudioChunk(0, chunkSamples, DspConstants.SAMPLE_RATE, chunkRms, System.currentTimeMillis()));
        } else if (totalLen > chunkSize && (totalLen % chunkSize) > (DspConstants.SAMPLE_RATE * 1.5f)) {
            // Jeśli została resztka > 1.5s, pobierz ostatnie 3.0s jako okno końcowe
            int start = totalLen - chunkSize;
            float[] chunkSamples = new float[chunkSize];
            System.arraycopy(samples, start, chunkSamples, 0, chunkSize);
            float chunkRms = AudioPreprocessor.computeRms(chunkSamples);
            chunks.add(new AudioChunk(windowIndex, chunkSamples, DspConstants.SAMPLE_RATE, chunkRms, System.currentTimeMillis()));
        }

        return chunks;
    }
}
