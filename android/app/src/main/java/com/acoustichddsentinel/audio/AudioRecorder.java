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
 * Obsługuje:
 * 1. Adaptacyjny nasłuch strumieniowy w stylu Shazam (Early Stopping & Multi-Window Voting)
 * 2. Próbkowanie 22050 Hz z antyaliasingową decymacją 2:1 w czasie rzeczywistym przy fallbacku 44100 Hz
 * 3. Analizę krokową 3-sekundowych okien z 50% zakładką (krok 1.5 s)
 * 4. Płynne raportowanie RMS oraz wczesne przerywanie po uzyskaniu pewnego wyniku.
 */
public class AudioRecorder {

    private static final String TAG = "AudioRecorder";

    public interface RecordCallback {
        void onProgress(int elapsedMs, int totalMs, float currentRms);
        void onRecordingComplete(List<AudioChunk> chunks, float overallRms);
        void onError(String message);
    }

    public interface AdaptiveStreamCallback {
        void onProgress(int elapsedMs, int maxMs, float currentRms);
        /**
         * Wywoływane natychmiast w locie, gdy w buforze audio dostępny jest kolejny pełny 3-sekundowy wycinek (66150 próbek).
         * @param windowIndex numer okna (0, 1, 2...)
         * @param chunk obiekt AudioChunk zawierający 66150 próbek przy 22050 Hz
         * @return true jeśli wnioskowanie jest jednoznaczne i należy natychmiast zatrzymać nagrywanie (Early Stopping), false jeśli słuchamy dalej
         */
        boolean onWindowReady(int windowIndex, AudioChunk chunk);
        void onStreamComplete(List<AudioChunk> evaluatedChunks, float overallRms, boolean stoppedEarly);
        void onError(String message);
    }

    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private Thread recordingThread;
    private AudioRecord audioRecord;

    public boolean isRecording() {
        return isRecording.get();
    }

    /**
     * Uruchamia adaptacyjny nasłuch wielookienkowy w stylu Shazam (od 3.0s do maxSeconds z krokiem 1.5s).
     *
     * @param maxSeconds maksymalny czas nasłuchu (np. 9 sekund)
     * @param callback   interfejs zdarzeń dla kolejnych okien i zakończenia strumienia
     */
    @SuppressLint("MissingPermission")
    public synchronized void startAdaptiveListening(int maxSeconds, AdaptiveStreamCallback callback) {
        // Zabezpieczenie przed wiszącym poprzednim wątkiem lub niezwolnionym deskryptorem mikrofonu
        if (isRecording.get() || (recordingThread != null && recordingThread.isAlive())) {
            Log.w(TAG, "Wykryto aktywną sesję lub wątek - wymuszanie bezpiecznego resetu przed nowym badaniem.");
            isRecording.set(false);
            stopInternal();
            if (recordingThread != null) {
                try {
                    recordingThread.join(400);
                } catch (InterruptedException ignored) {}
                recordingThread = null;
            }
            try {
                // Odczekanie 120 ms na zwolnienie sesji sprzętowej przez AudioFlinger / Android HAL
                Thread.sleep(120);
            } catch (InterruptedException ignored) {}
        }

        int targetSr = DspConstants.SAMPLE_RATE; // 22050 Hz
        int hardwareSr = targetSr;
        int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        int audioFormat = AudioFormat.ENCODING_PCM_16BIT;

        int minBufferSize = AudioRecord.getMinBufferSize(hardwareSr, channelConfig, audioFormat);
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
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

            int windowSize = DspConstants.CHUNK_SAMPLES; // 66150 próbek (3.0s)
            int step = windowSize / 2;                   // 33075 próbek (1.5s - 50% overlap)
            int maxWindows = 5;
            int maxCapacity = maxSeconds * targetSr + 16384;

            float[] sampleBuffer = new float[maxCapacity];
            int sampleCount = 0;

            short[] readBuffer = new short[bufferSize / 2];
            long startTime = System.currentTimeMillis();
            long maxDurationMs = maxSeconds * 1000L;

            List<AudioChunk> evaluatedChunks = new ArrayList<>();
            int nextWindowIndex = 0;
            boolean stoppedEarly = false;
            boolean hasLeftover = false;
            float leftover = 0.0f;

            while (isRecording.get() && (System.currentTimeMillis() - startTime) < maxDurationMs) {
                int readShorts = audioRecord.read(readBuffer, 0, readBuffer.length);
                if (readShorts > 0) {
                    float sumSq = 0.0f;

                    if (finalHardwareSr == DspConstants.FALLBACK_SAMPLE_RATE) {
                        // Decymacja 2:1 z 44100 Hz do 22050 Hz w czasie rzeczywistym
                        for (int i = 0; i < readShorts; i++) {
                            float s = readBuffer[i] / 32768.0f;
                            sumSq += s * s;
                            if (hasLeftover) {
                                if (sampleCount < maxCapacity) {
                                    sampleBuffer[sampleCount++] = 0.5f * (leftover + s);
                                }
                                hasLeftover = false;
                            } else {
                                leftover = s;
                                hasLeftover = true;
                            }
                        }
                    } else {
                        // Natywny sygnał 22050 Hz
                        for (int i = 0; i < readShorts; i++) {
                            float s = readBuffer[i] / 32768.0f;
                            sumSq += s * s;
                            if (sampleCount < maxCapacity) {
                                sampleBuffer[sampleCount++] = s;
                            }
                        }
                    }

                    float currentRms = (float) Math.sqrt(sumSq / readShorts);
                    int elapsedMs = (int) (System.currentTimeMillis() - startTime);
                    callback.onProgress(elapsedMs, (int) maxDurationMs, currentRms);

                    // Sprawdzenie, czy zgromadzono wystarczającą liczbę próbek dla kolejnego okna 3.0s
                    int neededSamples = windowSize + nextWindowIndex * step;
                    while (sampleCount >= neededSamples && isRecording.get() && nextWindowIndex < maxWindows) {
                        int start = nextWindowIndex * step;
                        float[] windowSamples = new float[windowSize];
                        System.arraycopy(sampleBuffer, start, windowSamples, 0, windowSize);
                        float windowRms = AudioPreprocessor.computeRms(windowSamples);
                        AudioChunk chunk = new AudioChunk(nextWindowIndex, windowSamples, targetSr, windowRms, System.currentTimeMillis());
                        evaluatedChunks.add(chunk);

                        boolean shouldStopEarly = callback.onWindowReady(nextWindowIndex, chunk);
                        nextWindowIndex++;

                        if (shouldStopEarly) {
                            stoppedEarly = true;
                            isRecording.set(false);
                            break;
                        }

                        if (nextWindowIndex >= maxWindows) {
                            isRecording.set(false);
                            break;
                        }

                        neededSamples = windowSize + nextWindowIndex * step;
                    }
                } else if (readShorts < 0) {
                    Log.w(TAG, "AudioRecord błąd odczytu próbek: " + readShorts);
                    break;
                }
            }

            // Zwolnienie sprzętu
            stopInternal();

            if (sampleCount == 0) {
                callback.onError("Nie zarejestrowano żadnych próbek audio.");
                return;
            }

            // Jeśli zatrzymano przed ukończeniem pierwszego pełnego okna 3.0s (np. ręczne przerwanie po 2.2s)
            if (evaluatedChunks.isEmpty()) {
                if (sampleCount >= targetSr * 2) {
                    float[] windowSamples = new float[windowSize];
                    System.arraycopy(sampleBuffer, 0, windowSamples, 0, Math.min(sampleCount, windowSize));
                    float windowRms = AudioPreprocessor.computeRms(windowSamples);
                    AudioChunk chunk = new AudioChunk(0, windowSamples, targetSr, windowRms, System.currentTimeMillis());
                    evaluatedChunks.add(chunk);
                    callback.onWindowReady(0, chunk);
                } else {
                    callback.onError("Nagranie zbyt krótkie (wymagane min. 2-3 sekundy do analizy).");
                    return;
                }
            }

            float overallRms = AudioPreprocessor.computeRms(sampleBuffer, sampleCount);
            callback.onStreamComplete(evaluatedChunks, overallRms, stoppedEarly);

        }, "AdaptiveAudioRecorderThread");

        recordingThread.start();
    }

    /**
     * Zgodność wsteczna: Uruchamia rejestrację na określony stały czas trwania.
     */
    public synchronized void startRecording(int durationSeconds, RecordCallback callback) {
        startAdaptiveListening(durationSeconds, new AdaptiveStreamCallback() {
            @Override
            public void onProgress(int elapsedMs, int maxMs, float currentRms) {
                callback.onProgress(elapsedMs, maxMs, currentRms);
            }

            @Override
            public boolean onWindowReady(int windowIndex, AudioChunk chunk) {
                // W trybie tradycyjnym nie przerywamy przedwcześnie
                return false;
            }

            @Override
            public void onStreamComplete(List<AudioChunk> evaluatedChunks, float overallRms, boolean stoppedEarly) {
                callback.onRecordingComplete(evaluatedChunks, overallRms);
            }

            @Override
            public void onError(String message) {
                callback.onError(message);
            }
        });
    }

    /**
     * Zatrzymuje bieżącą rejestrację audio (wywoływane np. przy ponownym stuknięciu w przycisk Hero).
     */
    public synchronized void stopRecording() {
        isRecording.set(false);
        stopInternal();
        if (recordingThread != null && recordingThread.isAlive()) {
            try {
                recordingThread.join(350);
            } catch (InterruptedException ignored) {}
            recordingThread = null;
        }
    }

    private synchronized void stopInternal() {
        isRecording.set(false);
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
}
