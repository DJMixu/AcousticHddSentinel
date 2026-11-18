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
import java.util.concurrent.locks.ReentrantLock;

/**
 * Rejestrator sygnału audio z mikrofonu sprzętowego smartfona (android.media.AudioRecord).
 * Obsługuje:
 * 1. Adaptacyjny nasłuch strumieniowy w stylu Shazam (Early Stopping & Multi-Window Voting).
 * 2. Próbkowanie 22050 Hz z antyaliasingową decymacją 2:1 w czasie rzeczywistym przy fallbacku 44100 Hz.
 * 3. Analizę krokową 3-sekundowych okien z 50% zakładką (krok 1.5 s).
 * 4. Płynne raportowanie RMS oraz wczesne przerywanie po uzyskaniu pewnego wyniku.
 * 5. Odporność na race condition i blokady AudioFlinger (Mutex Lock + Retry & Fallback Handler).
 */
public class AudioRecorder {

    private static final String TAG = "AudioRecorder";
    private static final int MAX_INIT_RETRIES = 3;

    // Współczynniki 2-sekcyjnego kaskadowego filtru Butterwortha 4. rzędu (fc = 9500 Hz przy fs = 44100 Hz)
    // Sekcja 1
    public static final float B0_1 = 0.058872744f;
    public static final float B1_1 = 0.11774549f;
    public static final float B2_1 = 0.058872744f;
    public static final float A1_1 = -0.22665857f;
    public static final float A2_1 = 0.051437586f;

    // Sekcja 2
    public static final float B0_2 = 1.0f;
    public static final float B1_2 = 2.0f;
    public static final float B2_2 = 1.0f;
    public static final float A1_2 = -0.31385655f;
    public static final float A2_2 = 0.4559369f;

    /**
     * Kaskadowy filtr Butterwortha 4. rzędu (Direct Form II Transposed) z decymacją 2:1
     * z 44100 Hz do 22050 Hz.
     */
    public static class ButterworthDecimator {
        private float s1_1 = 0.0f, s2_1 = 0.0f;
        private float s1_2 = 0.0f, s2_2 = 0.0f;
        private boolean decimateToggle = false;

        public void reset() {
            s1_1 = 0.0f;
            s2_1 = 0.0f;
            s1_2 = 0.0f;
            s2_2 = 0.0f;
            decimateToggle = false;
        }

        public float[] process(float[] input44k) {
            if (input44k == null || input44k.length == 0) {
                return new float[0];
            }
            float[] output22k = new float[input44k.length / 2];
            int outIdx = 0;
            for (float s : input44k) {
                // Kaskada Biquad Sekcja 1
                float y1 = B0_1 * s + s1_1;
                s1_1 = B1_1 * s - A1_1 * y1 + s2_1;
                s2_1 = B2_1 * s - A2_1 * y1;

                // Kaskada Biquad Sekcja 2
                float y2 = B0_2 * y1 + s1_2;
                s1_2 = B1_2 * y1 - A1_2 * y2 + s2_2;
                s2_2 = B2_2 * y1 - A2_2 * y2;

                decimateToggle = !decimateToggle;
                if (decimateToggle && outIdx < output22k.length) {
                    output22k[outIdx++] = y2;
                }
            }
            return output22k;
        }

        public float[] process(short[] inputShorts44k, int offset, int length) {
            if (inputShorts44k == null || length <= 0) {
                return new float[0];
            }
            float[] output22k = new float[length / 2];
            int outIdx = 0;
            for (int i = 0; i < length; i++) {
                float s = inputShorts44k[offset + i] / 32768.0f;

                // Kaskada Biquad Sekcja 1
                float y1 = B0_1 * s + s1_1;
                s1_1 = B1_1 * s - A1_1 * y1 + s2_1;
                s2_1 = B2_1 * s - A2_1 * y1;

                // Kaskada Biquad Sekcja 2
                float y2 = B0_2 * y1 + s1_2;
                s1_2 = B1_2 * y1 - A1_2 * y2 + s2_2;
                s2_2 = B2_2 * y1 - A2_2 * y2;

                decimateToggle = !decimateToggle;
                if (decimateToggle && outIdx < output22k.length) {
                    output22k[outIdx++] = y2;
                }
            }
            return output22k;
        }
    }

    public static float[] filterAndDecimate(float[] input44k) {
        return new ButterworthDecimator().process(input44k);
    }

    public static float[] filterAndDecimate(short[] inputShorts44k) {
        if (inputShorts44k == null) return new float[0];
        return new ButterworthDecimator().process(inputShorts44k, 0, inputShorts44k.length);
    }

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
    private final ReentrantLock recorderLock = new ReentrantLock();
    private Thread recordingThread;
    private AudioRecord audioRecord;

    public boolean isRecording() {
        return isRecording.get();
    }

    /**
     * Uruchamia adaptacyjny nasłuch wielookienkowy w stylu Shazam (od 3.0s do maxSeconds z krokiem 1.5s).
     * Wykorzystuje thread-safe mutex oraz Retry & Fallback Handler dla źródeł mikrofonowych.
     *
     * @param maxSeconds maksymalny czas nasłuchu (np. 9 sekund)
     * @param callback   interfejs zdarzeń dla kolejnych okien i zakończenia strumienia
     */
    @SuppressLint("MissingPermission")
    public void startAdaptiveListening(int maxSeconds, AdaptiveStreamCallback callback) {
        recorderLock.lock();
        try {
            // 1. Zabezpieczenie przed wiszącą sesją lub niezwolnionym deskryptorem mikrofonu
            if (isRecording.get() || (recordingThread != null && recordingThread.isAlive())) {
                Log.w(TAG, "Wykryto aktywną sesję lub wątek - wymuszanie bezpiecznego resetu przed nowym badaniem.");
                isRecording.set(false);
                stopInternal();

                if (recordingThread != null) {
                    try {
                        recordingThread.join(600);
                    } catch (InterruptedException ignored) {}
                    if (recordingThread.isAlive()) {
                        recordingThread.interrupt();
                    }
                    recordingThread = null;
                }

                try {
                    // Odczekanie 150 ms na zwolnienie sesji sprzętowej przez AudioFlinger / Android HAL
                    Thread.sleep(150);
                } catch (InterruptedException ignored) {}
            }

            int targetSr = DspConstants.SAMPLE_RATE; // 22050 Hz
            int channelConfig = AudioFormat.CHANNEL_IN_MONO;
            int audioFormat = AudioFormat.ENCODING_PCM_16BIT;

            // 2. Automatyczny mechanizm próby ponownego podłączenia (Retry & Fallback Handler)
            InitResult initResult = null;
            for (int attempt = 1; attempt <= MAX_INIT_RETRIES; attempt++) {
                initResult = tryInitializeAudioRecord(targetSr, channelConfig, audioFormat);
                if (initResult != null && initResult.record != null) {
                    break;
                }
                Log.w(TAG, "Próba inicjalizacji AudioRecord #" + attempt + " nie powiodła się. Ponawianie po pauzie...");
                try {
                    Thread.sleep(120L * attempt);
                } catch (InterruptedException ignored) {}
            }

            if (initResult == null || initResult.record == null) {
                callback.onError("Nie udało się uzyskać dostępu do mikrofonu po wielokrotnych próbach (błąd AudioFlinger HAL).");
                return;
            }

            this.audioRecord = initResult.record;
            final int finalHardwareSr = initResult.sampleRate;
            final int bufferSize = initResult.bufferSize;

            isRecording.set(true);

            recordingThread = new Thread(() -> {
                // Bezpieczny start nagrywania
                try {
                    audioRecord.startRecording();
                    if (audioRecord.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
                        throw new IllegalStateException("AudioRecord nie przeszedł w stan RECORDSTATE_RECORDING");
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Błąd startu nagrywania: " + e.getMessage(), e);
                    recorderLock.lock();
                    try {
                        isRecording.set(false);
                        stopInternal();
                    } finally {
                        recorderLock.unlock();
                    }
                    callback.onError("Błąd startu mikrofonu sprzętowego: " + e.getMessage());
                    return;
                }

                int windowSize = DspConstants.CHUNK_SAMPLES; // 66150 próbek (3.0s)
                int step = windowSize / 2;                   // 33075 próbek (1.5s - 50% overlap)
                int maxWindows = Math.max(1, (maxSeconds * 2) - 1);
                int maxCapacity = maxSeconds * targetSr + 16384;

                float[] sampleBuffer = new float[maxCapacity];
                int sampleCount = 0;

                short[] readBuffer = new short[bufferSize / 2];
                long startTime = System.currentTimeMillis();
                long maxDurationMs = maxSeconds * 1000L;

                List<AudioChunk> evaluatedChunks = new ArrayList<>();
                int nextWindowIndex = 0;
                boolean stoppedEarly = false;
                float s1_1 = 0.0f, s2_1 = 0.0f;
                float s1_2 = 0.0f, s2_2 = 0.0f;
                boolean decimateToggle = false;

                while (isRecording.get() && (System.currentTimeMillis() - startTime) < maxDurationMs) {
                    int readShorts = audioRecord.read(readBuffer, 0, readBuffer.length);
                    if (readShorts > 0) {
                        float sumSq = 0.0f;

                        if (finalHardwareSr == DspConstants.FALLBACK_SAMPLE_RATE) {
                            // Antyaliasingowa filtracja Butterwortha 4. rzędu + decymacja 2:1 z 44100 Hz do 22050 Hz
                            for (int i = 0; i < readShorts; i++) {
                                float s = readBuffer[i] / 32768.0f;
                                sumSq += s * s;

                                // Kaskada Biquad Sekcja 1
                                float y1 = B0_1 * s + s1_1;
                                s1_1 = B1_1 * s - A1_1 * y1 + s2_1;
                                s2_1 = B2_1 * s - A2_1 * y1;

                                // Kaskada Biquad Sekcja 2
                                float y2 = B0_2 * y1 + s1_2;
                                s1_2 = B1_2 * y1 - A1_2 * y2 + s2_2;
                                s2_2 = B2_2 * y1 - A2_2 * y2;

                                decimateToggle = !decimateToggle;
                                if (decimateToggle && sampleCount < maxCapacity) {
                                    sampleBuffer[sampleCount++] = y2;
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

                            // Automatyczne podbicie cichego sygnału (zgodne z potokiem przygotowania danych)
                            float[] processedSamples = AudioPreprocessor.checkAndBoostAudio(windowSamples, DspConstants.TARGET_BOOST_RMS);
                            float windowRms = AudioPreprocessor.computeRms(processedSamples);
                            AudioChunk chunk = new AudioChunk(nextWindowIndex, processedSamples, targetSr, windowRms, System.currentTimeMillis());
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
                    } else if (readShorts == AudioRecord.ERROR_INVALID_OPERATION || readShorts == AudioRecord.ERROR_BAD_VALUE) {
                        Log.w(TAG, "AudioRecord błąd odczytu próbek: " + readShorts);
                        break;
                    }
                }

                // Zwolnienie sprzętu
                recorderLock.lock();
                try {
                    stopInternal();
                } finally {
                    recorderLock.unlock();
                }

                if (sampleCount == 0) {
                    callback.onError("Nie zarejestrowano żadnych próbek audio.");
                    return;
                }

                // Jeśli zatrzymano przed ukończeniem pierwszego pełnego okna 3.0s (np. ręczne przerwanie)
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

        } finally {
            recorderLock.unlock();
        }
    }

    private static class InitResult {
        final AudioRecord record;
        final int sampleRate;
        final int bufferSize;

        InitResult(AudioRecord record, int sampleRate, int bufferSize) {
            this.record = record;
            this.sampleRate = sampleRate;
            this.bufferSize = bufferSize;
        }
    }

    /**
     * Próba utworzenia zainicjalizowanego obiektu AudioRecord z kaskadowym fallbackiem źródeł i częstotliwości próbkowania.
     */
    @SuppressLint("MissingPermission")
    private static InitResult tryInitializeAudioRecord(int targetSr, int channelConfig, int audioFormat) {
        int[] sampleRates = new int[] { targetSr, DspConstants.FALLBACK_SAMPLE_RATE };
        int[] audioSources = new int[] {
            MediaRecorder.AudioSource.UNPROCESSED,
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.MIC
        };

        for (int sr : sampleRates) {
            int minBuf = AudioRecord.getMinBufferSize(sr, channelConfig, audioFormat);
            if (minBuf <= 0) {
                continue;
            }
            int bufferSize = Math.max(minBuf * 2, 4096);

            for (int source : audioSources) {
                try {
                    AudioRecord record = new AudioRecord(source, sr, channelConfig, audioFormat, bufferSize);
                    if (record.getState() == AudioRecord.STATE_INITIALIZED) {
                        Log.i(TAG, "AudioRecord pomyślnie zainicjalizowany ze źródłem: " + getSourceName(source) + " (" + sr + " Hz)");
                        return new InitResult(record, sr, bufferSize);
                    }
                    record.release();
                } catch (Exception e) {
                    Log.w(TAG, "Inicjalizacja źródła " + getSourceName(source) + " (" + sr + " Hz) nie powiodła się: " + e.getMessage());
                }
            }
        }

        return null;
    }

    /**
     * Zgodność wsteczna: Uruchamia rejestrację na określony stały czas trwania.
     */
    public void startRecording(int durationSeconds, RecordCallback callback) {
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
    public void stopRecording() {
        recorderLock.lock();
        try {
            isRecording.set(false);
            stopInternal();
            if (recordingThread != null && recordingThread.isAlive()) {
                try {
                    recordingThread.join(450);
                } catch (InterruptedException ignored) {}
                if (recordingThread.isAlive()) {
                    recordingThread.interrupt();
                }
                recordingThread = null;
            }
        } finally {
            recorderLock.unlock();
        }
    }

    private void stopInternal() {
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

    private static String getSourceName(int source) {
        switch (source) {
            case MediaRecorder.AudioSource.UNPROCESSED: return "UNPROCESSED";
            case MediaRecorder.AudioSource.VOICE_RECOGNITION: return "VOICE_RECOGNITION";
            case MediaRecorder.AudioSource.MIC: return "MIC";
            default: return String.valueOf(source);
        }
    }
}
