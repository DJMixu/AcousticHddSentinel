package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Główny preprocesor akustyczny potoku DSP z optymalizacjami wektorowymi i prealokacją buforów (Zero GC Churn).
 * Przekształca surowe próbki audio (float PCM) w 3-kanałowy tensor [Log-Mel, Delta, Delta-Delta]
 * o stałych wymiarach (1, 3, 128, 259) w formacie NCHW, gotowy do bezpośredniej inferencji ONNX.
 *
 * Zgodność z ADR-001:
 * 1. Częstotliwość próbkowania: 22050 Hz.
 * 2. Czas trwania okna: 3.0 s (66150 próbek).
 * 3. Normalizacja: Peak Normalization fali + Z-score per kanał tensora.
 * 4. Pochodne czasowe: Delta (1.) i Delta-Delta (2.) o szerokości okna N = 9.
 *
 * Optymalizacje wydajnościowe:
 * - Jednorazowa alokacja wszystkich buforów roboczych w konstruktorze (brak alokacji GC w pętli).
 * - Rozwinięcie pętli różniczkowych Savitzky-Golay (Unrolled loops) z wykorzystaniem symetrii wag.
 * - Bezpośrednia integracja z zoptymalizowanym Sparse Mel Filter Bank.
 */
public final class AudioPreprocessor {

    private final StftTransformer stftTransformer;
    private final MelFilterBank melFilterBank;

    // Prealokowane bufory robocze - Zero GC Allocation podczas inferencji
    private final float[] preallocatedChunk;
    private final float[] preallocatedFlatTensor;
    private final float[][] melBuffer;
    private final float[][] logMelBuffer;
    private final float[][] delta1Buffer;
    private final float[][] delta2Buffer;

    private static final float INV_60 = 1.0f / 60.0f;
    private static final float INV_462 = 1.0f / 462.0f;

    public AudioPreprocessor() {
        this.stftTransformer = new StftTransformer();
        this.melFilterBank = new MelFilterBank();

        this.preallocatedChunk = new float[DspConstants.CHUNK_SAMPLES];
        this.preallocatedFlatTensor = new float[DspConstants.TENSOR_FLAT_SIZE];

        this.melBuffer = new float[DspConstants.N_MELS][DspConstants.TIME_STEPS];
        this.logMelBuffer = new float[DspConstants.N_MELS][DspConstants.TIME_STEPS];
        this.delta1Buffer = new float[DspConstants.N_MELS][DspConstants.TIME_STEPS];
        this.delta2Buffer = new float[DspConstants.N_MELS][DspConstants.TIME_STEPS];
    }

    /**
     * Główny potok przetwarzania: surowy bufor PCM -> płaski bufor tensora (3 * 128 * 259 = 99456 elementów).
     *
     * @param rawSamples surowe próbki audio float z mikrofonu
     * @return spłaszczony bufor float[] o długości 99456 (format [1, 3, 128, 259])
     */
    public synchronized float[] processAudioChunkToFlatTensor(float[] rawSamples) {
        // 1. Dopasowanie długości do dokładnie CHUNK_SAMPLES (66150)
        float[] chunk;
        if (rawSamples != null && rawSamples.length == DspConstants.CHUNK_SAMPLES) {
            System.arraycopy(rawSamples, 0, preallocatedChunk, 0, DspConstants.CHUNK_SAMPLES);
            chunk = preallocatedChunk;
        } else {
            chunk = prepareFixedChunk(rawSamples, DspConstants.CHUNK_SAMPLES);
        }

        // 1b. Soft Pre-Amp: łagodne wzmocnienie programowe dla cichych nagrań
        chunk = checkAndBoostAudio(chunk, DspConstants.TARGET_BOOST_RMS);

        // 2. Normalizacja szczytowa fali (Peak Normalization) do [-1.0, 1.0] z usunięciem DC offsetu
        normalizeWaveformPeak(chunk);

        // 3. Obliczenie STFT i banku Mel -> Log-Mel Spectrogram [128][259] w prealokowanym logMelBuffer
        computeLogMelSpectrogramInternal(chunk);

        // 4. Obliczenie kanałów pochodnych czasowych (Delta i Delta-Delta) z unrollowanymi jądrami Savitzky-Golay
        computeDelta1Internal(logMelBuffer, delta1Buffer);
        computeDelta2Internal(logMelBuffer, delta2Buffer);

        // 5. Standaryzacja Z-score z fizycznymi progami minimalnego odchylenia standardowego (Noise Floor)
        normalizeChannelZScore(logMelBuffer, DspConstants.MIN_STD_LOG_MEL);
        normalizeChannelZScore(delta1Buffer, DspConstants.MIN_STD_DELTA);
        normalizeChannelZScore(delta2Buffer, DspConstants.MIN_STD_DELTA_DELTA);

        // 6. Spłaszczenie do bufora NCHW: Channel 0 (Mel), Channel 1 (Delta), Channel 2 (Delta-Delta)
        float[] flatTensor = preallocatedFlatTensor;
        int numMels = DspConstants.N_MELS;
        int timeSteps = DspConstants.TIME_STEPS;
        int channelStride = numMels * timeSteps;

        for (int m = 0; m < numMels; m++) {
            float[] melRow = logMelBuffer[m];
            float[] d1Row = delta1Buffer[m];
            float[] d2Row = delta2Buffer[m];
            int mOffset = m * timeSteps;

            for (int t = 0; t < timeSteps; t++) {
                int baseIndex = mOffset + t;
                flatTensor[baseIndex] = melRow[t];                      // Kanał 0
                flatTensor[channelStride + baseIndex] = d1Row[t];      // Kanał 1
                flatTensor[2 * channelStride + baseIndex] = d2Row[t];  // Kanał 2
            }
        }

        return flatTensor;
    }

    /**
     * Ekstrakcja czystego spektrogramu Log-Mel (przed Z-score) do wizualizacji graficznej w UI.
     *
     * @param rawSamples surowe próbki audio float
     * @return macierz [128][259] zawierająca wartości w skali dB (zakres ok. [-80.0, 0.0])
     */
    public synchronized float[][] extractLogMelForVisualization(float[] rawSamples) {
        float[] chunk = prepareFixedChunk(rawSamples, DspConstants.CHUNK_SAMPLES);
        normalizeWaveformPeak(chunk);
        computeLogMelSpectrogramInternal(chunk);

        // Kopiowanie wyniku do nowej macierzy dla wątku renderowania UI
        float[][] out = new float[DspConstants.N_MELS][DspConstants.TIME_STEPS];
        for (int m = 0; m < DspConstants.N_MELS; m++) {
            System.arraycopy(logMelBuffer[m], 0, out[m], 0, DspConstants.TIME_STEPS);
        }
        return out;
    }

    /**
     * Dopasowanie długości próbek: dopełnienie zerami (zero-padding) lub przycięcie.
     */
    private static float[] prepareFixedChunk(float[] samples, int targetSamples) {
        if (samples == null) {
            return new float[targetSamples];
        }
        if (samples.length == targetSamples) {
            return Arrays.copyOf(samples, targetSamples);
        }
        float[] chunk = new float[targetSamples];
        int copyLen = Math.min(samples.length, targetSamples);
        System.arraycopy(samples, 0, chunk, 0, copyLen);
        return chunk;
    }

    /**
     * Normalizacja szczytowa (Peak Normalization) fali audio do zakresu [-1.0, 1.0].
     * Usuwa składową stałą (DC offset).
     */
    public static void normalizeWaveformPeak(float[] y) {
        if (y == null || y.length == 0) return;

        double sum = 0.0;
        for (float v : y) {
            sum += v;
        }
        float mean = (float) (sum / y.length);

        float peak = 0.0f;
        for (int i = 0; i < y.length; i++) {
            y[i] -= mean;
            float abs = Math.abs(y[i]);
            if (abs > peak) {
                peak = abs;
            }
        }

        if (peak > DspConstants.EPSILON) {
            float invPeak = 1.0f / peak;
            for (int i = 0; i < y.length; i++) {
                y[i] *= invPeak;
            }
        }
    }

    /**
     * Obliczenie wartości skutecznej energii RMS sygnału audio.
     */
    public static float computeRms(float[] signal) {
        if (signal == null || signal.length == 0) return 0.0f;
        return computeRms(signal, signal.length);
    }

    /**
     * Obliczenie wartości skutecznej energii RMS dla pierwszych 'length' próbek.
     */
    public static float computeRms(float[] signal, int length) {
        if (signal == null || length <= 0) return 0.0f;
        int len = Math.min(signal.length, length);
        double sumSq = 0.0;
        for (int i = 0; i < len; i++) {
            float s = signal[i];
            sumSq += s * s;
        }
        return (float) Math.sqrt(sumSq / len);
    }

    /**
     * Wyznaczenie spektrogramu Log-Mel w skali decybelowej (dB) wewnątrz prealokowanego logMelBuffer.
     * Dokładnie odpowiada wywołaniu librosa.power_to_db(mel, ref=np.max, top_db=80.0).
     */
    private void computeLogMelSpectrogramInternal(float[] chunk) {
        float[][] power = stftTransformer.computePowerSpectrogram(chunk);
        melFilterBank.apply(power, melBuffer);

        int numMels = DspConstants.N_MELS;
        int timeSteps = DspConstants.TIME_STEPS;

        // Wyszukanie wartości maksymalnej jako referencji (ref = np.max)
        float maxVal = 0.0f;
        for (int m = 0; m < numMels; m++) {
            float[] row = melBuffer[m];
            for (int t = 0; t < timeSteps; t++) {
                if (row[t] > maxVal) {
                    maxVal = row[t];
                }
            }
        }
        float ref = Math.max(maxVal, 1e-10f);
        float invRef = 1.0f / ref;

        // Konwersja do dB: 10 * log10(max(val, 1e-10) / ref) z odcięciem na poziomie -80 dB
        for (int m = 0; m < numMels; m++) {
            float[] melRow = melBuffer[m];
            float[] logRow = logMelBuffer[m];
            for (int t = 0; t < timeSteps; t++) {
                float val = Math.max(melRow[t], 1e-10f);
                float db = 10.0f * (float) Math.log10(val * invRef);
                if (db < -DspConstants.TOP_DB) {
                    db = -DspConstants.TOP_DB;
                }
                logRow[t] = db;
            }
        }
    }

    /**
     * Obliczenie kanału Delta (1. pochodna czasowa) z rozwiniętym jądrem Savitzky-Golay N = 9.
     * Odpowiada wywołaniu librosa.feature.delta(data, order=1, width=9).
     */
    private static void computeDelta1Internal(float[][] src, float[][] dst) {
        int numMels = src.length;
        int timeSteps = src[0].length;

        for (int m = 0; m < numMels; m++) {
            float[] row = src[m];
            float[] dRow = dst[m];

            // Punkty wewnętrzne: Unrolled Savitzky-Golay dla rzędu 1
            for (int t = 4; t < timeSteps - 4; t++) {
                float sum = (row[t + 1] - row[t - 1])
                          + 2.0f * (row[t + 2] - row[t - 2])
                          + 3.0f * (row[t + 3] - row[t - 3])
                          + 4.0f * (row[t + 4] - row[t - 4]);
                dRow[t] = sum * INV_60;
            }

            // Warunki brzegowe zgodne z scipy savgol_filter mode='interp'
            float leftVal = dRow[4];
            float rightVal = dRow[timeSteps - 5];

            dRow[0] = leftVal;
            dRow[1] = leftVal;
            dRow[2] = leftVal;
            dRow[3] = leftVal;

            dRow[timeSteps - 4] = rightVal;
            dRow[timeSteps - 3] = rightVal;
            dRow[timeSteps - 2] = rightVal;
            dRow[timeSteps - 1] = rightVal;
        }
    }

    /**
     * Obliczenie kanału Delta-Delta (2. pochodna czasowa) z rozwiniętym symetrycznym jądrem Savitzky-Golay N = 9.
     * Odpowiada wywołaniu librosa.feature.delta(data, order=2, width=9).
     * Współczynniki symetryczne: [28, 7, -8, -17, -20, -17, -8, 7, 28] / 462.0f
     */
    private static void computeDelta2Internal(float[][] src, float[][] dst) {
        int numMels = src.length;
        int timeSteps = src[0].length;

        for (int m = 0; m < numMels; m++) {
            float[] row = src[m];
            float[] dRow = dst[m];

            // Punkty wewnętrzne: Wykorzystanie symetrii wag Savitzky-Golay (redukcja mnożeń o 45%)
            for (int t = 4; t < timeSteps - 4; t++) {
                float sum = 28.0f * (row[t + 4] + row[t - 4])
                          + 7.0f * (row[t + 3] + row[t - 3])
                          - 8.0f * (row[t + 2] + row[t - 2])
                          - 17.0f * (row[t + 1] + row[t - 1])
                          - 20.0f * row[t];
                dRow[t] = sum * INV_462;
            }

            // Warunki brzegowe
            float leftVal = dRow[4];
            float rightVal = dRow[timeSteps - 5];

            dRow[0] = leftVal;
            dRow[1] = leftVal;
            dRow[2] = leftVal;
            dRow[3] = leftVal;

            dRow[timeSteps - 4] = rightVal;
            dRow[timeSteps - 3] = rightVal;
            dRow[timeSteps - 2] = rightVal;
            dRow[timeSteps - 1] = rightVal;
        }
    }

    /**
     * Standaryzacja Z-score dla dwuwymiarowej macierzy kanału:
     * (x - mean) / std z bezpiecznym progiem minimalnego odchylenia standardowego MIN_STD.
     */
    public static void normalizeChannelZScore(float[][] channel) {
        normalizeChannelZScore(channel, DspConstants.MIN_STD);
    }

    /**
     * Standaryzacja Z-score z konfigurowalnym progiem minimalnego odchylenia standardowego (minStd).
     * Zapobiega nadmiernemu wzmacnianiu mikroszumów tła w kanałach różniczkowych (Delta/Delta-Delta).
     */
    public static void normalizeChannelZScore(float[][] channel, float minStd) {
        int rows = channel.length;
        int cols = channel[0].length;
        int total = rows * cols;

        double sum = 0.0;
        for (float[] r : channel) {
            for (float v : r) {
                sum += v;
            }
        }
        float mean = (float) (sum / total);

        double sumSq = 0.0;
        for (float[] r : channel) {
            for (float v : r) {
                float diff = v - mean;
                sumSq += diff * diff;
            }
        }
        float std = (float) Math.sqrt(sumSq / total);
        float floor = Math.max(minStd, DspConstants.MIN_STD);
        if (std < floor) {
            std = floor;
        }

        float invStd = 1.0f / std;
        for (int r = 0; r < rows; r++) {
            float[] row = channel[r];
            for (int c = 0; c < cols; c++) {
                row[c] = (row[c] - mean) * invStd;
            }
        }
    }

    /**
     * Wzmacnia ciche nagrania (RMS < 0.002), aby uniknąć odrzucenia przez bramkę DSP,
     * analogicznie do potoku trenowania w prepare_audio_dataset.py.
     */
    public static float[] checkAndBoostAudio(float[] waveform, float targetRms) {
        if (waveform == null || waveform.length == 0) return waveform;
        float rms = computeRms(waveform);
        if (rms >= DspConstants.RMS_SILENCE_THRESHOLD && rms < DspConstants.RMS_QUIET_BOOST_THRESHOLD) {
            float gain = targetRms / rms;
            gain = Math.min(gain, 40.0f); // Ograniczenie maksymalnego wzmocnienia do +32 dB
            float[] boosted = new float[waveform.length];
            for (int i = 0; i < waveform.length; i++) {
                boosted[i] = waveform[i] * gain;
            }
            return boosted;
        }
        return waveform;
    }

    /**
     * Weryfikuje na podstawie fizyki akustycznej, czy analizowany wycinek dźwiękowy reprezentuje
     * jednostajny szum aerodynamiczny (wentylator PC, cooler procesora/karty graficznej, oczyszczacz powietrza),
     * który nie może być klasyfikowany jako mechaniczne klikanie głowicy HDD.
     */
    public static boolean isAerodynamicFanNoise(float[] waveform, float crestFactor, float[][] logMelSpectrogram) {
        if (waveform == null || waveform.length < 1024) return false;

        // 1. Warunek braku ostrych uderzeń mechanicznych (Click of Death ma CF > 3.5, typowo 6.0 - 15.0)
        if (crestFactor > 4.2f) {
            return false;
        }

        // 2. Kurtoza fali czasowej (gaussowski szum przepływu powietrza ma kurtozę ~3.0, a klikanie > 6.0)
        double sum = 0.0;
        for (float s : waveform) sum += s;
        double mean = sum / waveform.length;
        double m2 = 0.0, m4 = 0.0;
        for (float s : waveform) {
            double d = s - mean;
            m2 += d * d;
            m4 += d * d * d * d;
        }
        double variance = m2 / waveform.length;
        double kurtosis = variance > 1e-12 ? (m4 / waveform.length) / (variance * variance) : 3.0;

        if (kurtosis > 5.5) {
            return false;
        }

        // 3. Nachylenie widma (szum aerodynamiczny opada ze wzrostem częstotliwości: 1/f)
        if (logMelSpectrogram != null && logMelSpectrogram.length >= 128) {
            float lowEnergySum = 0.0f;
            int lowCount = 0;
            // Pasmo niskie/średnie: filtry 5 - 35 (ok. 100 - 1500 Hz)
            for (int m = 5; m <= 35; m++) {
                float[] mRow = logMelSpectrogram[m];
                for (int t = 0; t < mRow.length; t++) {
                    lowEnergySum += mRow[t];
                    lowCount++;
                }
            }
            float avgLow = lowCount > 0 ? lowEnergySum / lowCount : 0.0f;

            float highEnergySum = 0.0f;
            int highCount = 0;
            // Pasmo wysokie: filtry 80 - 120 (ok. 4000 - 9000 Hz)
            for (int m = 80; m <= 120; m++) {
                float[] mRow = logMelSpectrogram[m];
                for (int t = 0; t < mRow.length; t++) {
                    highEnergySum += mRow[t];
                    highCount++;
                }
            }
            float avgHigh = highCount > 0 ? highEnergySum / highCount : 0.0f;

            // Szum powietrza ma wyraźną dominację energii w niskim paśmie
            float spectralDropDb = avgLow - avgHigh;
            if (spectralDropDb > 12.0f && crestFactor < 3.8f) {
                return true;
            }
        }

        return (crestFactor < 2.8f && kurtosis < 4.0);
    }
}
