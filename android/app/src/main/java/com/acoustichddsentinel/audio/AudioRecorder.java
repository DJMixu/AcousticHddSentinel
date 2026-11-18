package com.acoustichddsentinel.audio;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
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
 * 3. Opcjonalny tryb czystego toru RAW (AudioSource.UNPROCESSED + decymacja) omijający systemowe DSP.
 * 4. Odpinanie i wyłączanie sprzętowych filtrów systemowych (NoiseSuppressor, AcousticEchoCanceler, AGC)
 *    dla zachowania niskich harmonicznych rotacji silnika wrzeciona (5400/7200 RPM) oraz transjentów stukania.
 * 5. Odporność na race condition i blokady AudioFlinger (Mutex Lock + Retry & Fallback Handler).
 *
 * Zgodność z ADR-001 i ADR-002 (Czysta Java 17, Zero Kotlin).
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
        boolean onWindowReady(int windowIndex, AudioChunk chunk);
        void onStreamComplete(List<AudioChunk> evaluatedChunks, float overallRms, boolean stoppedEarly);
        void onError(String message);
    }

    /**
     * Kontroler efektów sprzętowych audiofx (odpinanie filtrów mowy i AGC w trybie RAW).
     */
    private static class AudioFxController {
        private NoiseSuppressor noiseSuppressor;
        private AutomaticGainControl automaticGainControl;
        private AcousticEchoCanceler acousticEchoCanceler;

        void attachAndDisable(int audioSessionId) {
            try {
                if (NoiseSuppressor.isAvailable()) {
                    noiseSuppressor = NoiseSuppressor.create(audioSessionId);
                    if (noiseSuppressor != null) {
                        noiseSuppressor.setEnabled(false);
                        Log.i(TAG, "AudioFx: NoiseSuppressor wyłączony dla sesji " + audioSessionId);
                    }
                }
            } catch (Throwable t) {
                Log.w(TAG, "AudioFx: Pominięto NoiseSuppressor: " + t.getMessage());
            }

            try {
                if (AutomaticGainControl.isAvailable()) {
                    automaticGainControl = AutomaticGainControl.create(audioSessionId);
                    if (automaticGainControl != null) {
                        automaticGainControl.setEnabled(false);
                        Log.i(TAG, "AudioFx: AutomaticGainControl wyłączony dla sesji " + audioSessionId);
                    }
                }
            } catch (Throwable t) {
                Log.w(TAG, "AudioFx: Pominięto AutomaticGainControl: " + t.getMessage());
            }

            try {
                if (AcousticEchoCanceler.isAvailable()) {
                    acousticEchoCanceler = AcousticEchoCanceler.create(audioSessionId);
                    if (acousticEchoCanceler != null) {
                        acousticEchoCanceler.setEnabled(false);
                        Log.i(TAG, "AudioFx: AcousticEchoCanceler wyłączony dla sesji " + audioSessionId);
                    }
                }
            } catch (Throwable t) {
                Log.w(TAG, "AudioFx: Pominięto AcousticEchoCanceler: " + t.getMessage());
            }
        }

        void release() {
            if (noiseSuppressor != null) {
                try { noiseSuppressor.release(); } catch (Throwable ignored) {}
                noiseSuppressor = null;
            }
            if (automaticGainControl != null) {
                try { automaticGainControl.release(); } catch (Throwable ignored) {}
                automaticGainControl = null;
            }
            if (acousticEchoCanceler != null) {
                try { acousticEchoCanceler.release(); } catch (Throwable ignored) {}
                acousticEchoCanceler = null;
            }
        }
    }

    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    private final ReentrantLock recorderLock = new ReentrantLock();
    private final AudioFxController audioFxController = new AudioFxController();

    private Thread recordingThread;
    private AudioRecord audioRecord;
    private volatile boolean rawAudioMode = AudioPreferences.DEFAULT_RAW_AUDIO_MODE;

    public boolean isRecording() {
        return isRecording.get();
    }

    public boolean isRawAudioMode() {
        return rawAudioMode;
    }

    /**
     * Włącza lub wyłącza opcjonalny tryb czystego toru audio RAW.
     *
     * @param rawAudioMode true: priorytet UNPROCESSED + decymacja i wyłączanie filtrów mowy/AGC
     *                     false: tryb standardowy zgodności
     */
    public void setRawAudioMode(boolean rawAudioMode) {
        this.rawAudioMode = rawAudioMode;
        Log.i(TAG, "Zmieniono tryb toru audio: rawAudioMode = " + rawAudioMode);
    }

    /**
     * Uruchamia adaptacyjny nasłuch wielookienkowy w stylu Shazam (od 3.0s do maxSeconds z krokiem 1.5s).
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

            // 2. Automatyczny mechanizm próby ponownego podłączenia z uwzględnieniem trybu RAW
            InitResult initResult = null;
            for (int attempt = 1; attempt <= MAX_INIT_RETRIES; attempt++) {
                initResult = tryInitializeAudioRecord(targetSr, channelConfig, audioFormat, this.rawAudioMode);
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

            // W trybie RAW zdejmujemy filtry systemowe z sesji AudioRecord
            if (this.rawAudioMode) {
                audioFxController.attachAndDisable(this.audioRecord.getAudioSessionId());
            }

            isRecording.set(true);

            recordingThread = new Thread(() -> {
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

                try {
                    while (isRecording.get()) {
                        long elapsedMs = System.currentTimeMillis() - startTime;
                        if (elapsedMs >= maxDurationMs) {
                            Log.i(TAG, "Osiągnięto limit czasu nagrywania (" + maxSeconds + "s). Zatrzymywanie nasłuchu.");
                            break;
                        }

                        int readShorts = audioRecord.read(readBuffer, 0, readBuffer.length);
                        if (readShorts <= 0) {
                            if (readShorts == AudioRecord.ERROR_INVALID_OPERATION || readShorts == AudioRecord.ERROR_DEAD_OBJECT) {
                                Log.e(TAG, "Krytyczny błąd odczytu z mikrofonu HAL: " + readShorts);
                                break;
                            }
                            try {
                                Thread.sleep(10);
                            } catch (InterruptedException e) {
                                break;
                            }
                            continue;
                        }

                        if (finalHardwareSr == 22050) {
                            for (int i = 0; i < readShorts; i++) {
                                if (sampleCount < maxCapacity) {
                                    sampleBuffer[sampleCount++] = readBuffer[i] / 32768.0f;
                                }
                            }
                        } else {
                            // Decymacja 44100 Hz -> 22050 Hz filtrem Butterwortha 4. rzędu
                            for (int i = 0; i < readShorts; i++) {
                                float s = readBuffer[i] / 32768.0f;

                                float y1 = B0_1 * s + s1_1;
                                s1_1 = B1_1 * s - A1_1 * y1 + s2_1;
                                s2_1 = B2_1 * s - A2_1 * y1;

                                float y2 = B0_2 * y1 + s1_2;
                                s1_2 = B1_2 * y1 - A1_2 * y2 + s2_2;
                                s2_2 = B2_2 * y1 - A2_2 * y2;

                                decimateToggle = !decimateToggle;
                                if (decimateToggle && sampleCount < maxCapacity) {
                                    sampleBuffer[sampleCount++] = y2;
                                }
                            }
                        }

                        // Raportowanie poziomu RMS
                        if (sampleCount > 0) {
                            int rmsWindow = Math.min(sampleCount, 2048);
                            double sumSq = 0.0;
                            for (int i = sampleCount - rmsWindow; i < sampleCount; i++) {
                                sumSq += sampleBuffer[i] * sampleBuffer[i];
                            }
                            float currentRms = (float) Math.sqrt(sumSq / rmsWindow);
                            callback.onProgress((int) elapsedMs, (int) maxDurationMs, currentRms);
                        }

                        // Wnioskowanie wielookienkowe
                        while (sampleCount >= (nextWindowIndex * step + windowSize) && nextWindowIndex < maxWindows) {
                            int windowOffset = nextWindowIndex * step;
                            float[] windowSamples = new float[windowSize];
                            System.arraycopy(sampleBuffer, windowOffset, windowSamples, 0, windowSize);

                            float[] processedSamples = AudioPreprocessor.checkAndBoostAudio(windowSamples, DspConstants.TARGET_BOOST_RMS);
                            float windowRms = AudioPreprocessor.computeRms(processedSamples);
                            AudioChunk chunk = new AudioChunk(nextWindowIndex, processedSamples, targetSr, windowRms, System.currentTimeMillis());
                            evaluatedChunks.add(chunk);

                            boolean shouldStop = callback.onWindowReady(nextWindowIndex, chunk);
                            nextWindowIndex++;

                            if (shouldStop) {
                                stoppedEarly = true;
                                isRecording.set(false);
                                break;
                            }
                        }

                        if (stoppedEarly || nextWindowIndex >= maxWindows) {
                            break;
                        }
                    }
                } finally {
                    recorderLock.lock();
                    try {
                        stopInternal();
                    } finally {
                        recorderLock.unlock();
                    }

                    double totalSumSq = 0.0;
                    for (int i = 0; i < sampleCount; i++) {
                        totalSumSq += sampleBuffer[i] * sampleBuffer[i];
                    }
                    float overallRms = sampleCount > 0 ? (float) Math.sqrt(totalSumSq / sampleCount) : 0.0f;

                    callback.onStreamComplete(evaluatedChunks, overallRms, stoppedEarly);
                }
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

    private static class SourceSrPair {
        final int source;
        final int sampleRate;

        SourceSrPair(int source, int sampleRate) {
            this.source = source;
            this.sampleRate = sampleRate;
        }
    }

    /**
     * Próba utworzenia zainicjalizowanego obiektu AudioRecord z kaskadowym fallbackiem źródeł.
     * W trybie RAW: priorytetyzuje UNPROCESSED (22.05 kHz -> 44.1 kHz z decymacją),
     * eliminując inwersję priorytetów i omijając filtry mowy.
     */
    @SuppressLint("MissingPermission")
    private static InitResult tryInitializeAudioRecord(int targetSr, int channelConfig, int audioFormat, boolean rawAudioMode) {
        List<SourceSrPair> candidates = new ArrayList<>();

        if (rawAudioMode) {
            // Tryb czystego toru RAW: w pierwszej kolejności UNPROCESSED na obu próbkowaniach
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.UNPROCESSED, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.UNPROCESSED, DspConstants.FALLBACK_SAMPLE_RATE));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.VOICE_RECOGNITION, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.VOICE_RECOGNITION, DspConstants.FALLBACK_SAMPLE_RATE));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.MIC, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.MIC, DspConstants.FALLBACK_SAMPLE_RATE));
        } else {
            // Standardowy tryb kompatybilny
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.MIC, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.VOICE_RECOGNITION, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.MIC, DspConstants.FALLBACK_SAMPLE_RATE));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.VOICE_RECOGNITION, DspConstants.FALLBACK_SAMPLE_RATE));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.UNPROCESSED, targetSr));
            candidates.add(new SourceSrPair(MediaRecorder.AudioSource.UNPROCESSED, DspConstants.FALLBACK_SAMPLE_RATE));
        }

        for (SourceSrPair pair : candidates) {
            int minBuf = AudioRecord.getMinBufferSize(pair.sampleRate, channelConfig, audioFormat);
            if (minBuf <= 0) {
                continue;
            }
            int bufferSize = Math.max(minBuf * 2, 4096);

            try {
                AudioRecord record = new AudioRecord(pair.source, pair.sampleRate, channelConfig, audioFormat, bufferSize);
                if (record.getState() == AudioRecord.STATE_INITIALIZED) {
                    Log.i(TAG, "AudioRecord pomyślnie zainicjalizowany ze źródłem: " +
                            getSourceName(pair.source) + " (" + pair.sampleRate + " Hz, rawMode=" + rawAudioMode + ")");
                    return new InitResult(record, pair.sampleRate, bufferSize);
                }
                record.release();
            } catch (Exception e) {
                Log.w(TAG, "Inicjalizacja źródła " + getSourceName(pair.source) +
                        " (" + pair.sampleRate + " Hz) nie powiodła się: " + e.getMessage());
            }
        }

        return null;
    }

    public void startRecording(int durationSeconds, RecordCallback callback) {
        startAdaptiveListening(durationSeconds, new AdaptiveStreamCallback() {
            @Override
            public void onProgress(int elapsedMs, int maxMs, float currentRms) {
                callback.onProgress(elapsedMs, maxMs, currentRms);
            }

            @Override
            public boolean onWindowReady(int windowIndex, AudioChunk chunk) {
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
     * Zatrzymuje bieżącą rejestrację audio z bezpiecznym zwolnieniem blokady wątku UI.
     */
    public void stopRecording() {
        Thread threadToJoin = null;
        recorderLock.lock();
        try {
            isRecording.set(false);
            stopInternal();
            if (recordingThread != null && recordingThread.isAlive()) {
                threadToJoin = recordingThread;
                recordingThread = null;
            }
        } finally {
            recorderLock.unlock();
        }

        // join() wywołany poza sekcją krytyczną recorderLock (brak ryzyka deadlocka i laga w UI)
        if (threadToJoin != null) {
            try {
                threadToJoin.join(450);
            } catch (InterruptedException ignored) {}
            if (threadToJoin.isAlive()) {
                threadToJoin.interrupt();
            }
        }
    }

    private void stopInternal() {
        isRecording.set(false);
        audioFxController.release();
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
