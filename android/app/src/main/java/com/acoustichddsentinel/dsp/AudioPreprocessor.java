package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Główny preprocesor akustyczny potoku DSP.
 * Przekształca surowe próbki audio (float PCM) w 3-kanałowy tensor [Log-Mel, Delta, Delta-Delta]
 * o stałych wymiarach (1, 3, 128, 259) w formacie NCHW, gotowy do bezpośredniej inferencji ONNX.
 *
 * Zgodność z ADR-001:
 * 1. Częstotliwość próbkowania: 22050 Hz.
 * 2. Czas trwania okna: 3.0 s (66150 próbek).
 * 3. Normalizacja: Peak Normalization fali + Z-score per kanał tensora.
 * 4. Pochodne czasowe: Delta (1.) i Delta-Delta (2.) o szerokości okna N = 9.
 */
public final class AudioPreprocessor {

    private final StftTransformer stftTransformer;
    private final MelFilterBank melFilterBank;

    // Współczynniki jądra Savitzky-Golay dla 2. pochodnej (wielomian st. 2, szerokość 9)
    // Numeratory: [28, 7, -8, -17, -20, -17, -8, 7, 28] / 462.0
    private static final float[] SAVGOL_C2_KERNEL = {
        28.0f / 462.0f,
        7.0f / 462.0f,
        -8.0f / 462.0f,
        -17.0f / 462.0f,
        -20.0f / 462.0f,
        -17.0f / 462.0f,
        -8.0f / 462.0f,
        7.0f / 462.0f,
        28.0f / 462.0f
    };

    public AudioPreprocessor() {
        this.stftTransformer = new StftTransformer();
        this.melFilterBank = new MelFilterBank();
    }

    /**
     * Główny potok przetwarzania: surowy bufor PCM -> płaski bufor tensora (3 * 128 * 259 = 99456 elementów).
     *
     * @param rawSamples surowe próbki audio float z mikrofonu
     * @return spłaszczony bufor float[] o długości 99456 (format [1, 3, 128, 259])
     */
    public float[] processAudioChunkToFlatTensor(float[] rawSamples) {
        // 1. Dopasowanie długości do dokładnie CHUNK_SAMPLES (66150)
        float[] chunk = prepareFixedChunk(rawSamples, DspConstants.CHUNK_SAMPLES);

        // 2. Normalizacja szczytowa fali (Peak Normalization) do [-1.0, 1.0] z usunięciem DC offsetu
        normalizeWaveformPeak(chunk);

        // 3. Obliczenie STFT i banku Mel -> Log-Mel Spectrogram [128][259]
        float[][] logMel = computeLogMelSpectrogram(chunk);

        // 4. Obliczenie kanałów pochodnych czasowych (Delta i Delta-Delta)
        float[][] delta1 = computeDelta1(logMel);
        float[][] delta2 = computeDelta2(logMel);

        // 5. Standaryzacja Z-score osobno dla każdego kanału
        normalizeChannelZScore(logMel);
        normalizeChannelZScore(delta1);
        normalizeChannelZScore(delta2);

        // 6. Spłaszczenie do bufora NCHW: Channel 0 (Mel), Channel 1 (Delta), Channel 2 (Delta-Delta)
        float[] flatTensor = new float[DspConstants.TENSOR_FLAT_SIZE];
        int numMels = DspConstants.N_MELS;
        int timeSteps = DspConstants.TIME_STEPS;
        int channelStride = numMels * timeSteps;

        for (int m = 0; m < numMels; m++) {
            for (int t = 0; t < timeSteps; t++) {
                int baseIndex = m * timeSteps + t;
                flatTensor[baseIndex] = logMel[m][t];                       // Kanał 0
                flatTensor[channelStride + baseIndex] = delta1[m][t];       // Kanał 1
                flatTensor[2 * channelStride + baseIndex] = delta2[m][t];   // Kanał 2
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
    public float[][] extractLogMelForVisualization(float[] rawSamples) {
        float[] chunk = prepareFixedChunk(rawSamples, DspConstants.CHUNK_SAMPLES);
        normalizeWaveformPeak(chunk);
        return computeLogMelSpectrogram(chunk);
    }

    /**
     * Dopasowanie długości próbek: dopełnienie zerami (zero-padding) lub przycięcie.
     */
    private static float[] prepareFixedChunk(float[] samples, int targetSamples) {
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
        if (y.length == 0) return;

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
            for (int i = 0; i < y.length; i++) {
                y[i] /= peak;
            }
        }
    }

    /**
     * Obliczenie wartości skutecznej energii RMS sygnału audio.
     */
    public static float computeRms(float[] signal) {
        if (signal == null || signal.length == 0) return 0.0f;
        double sumSq = 0.0;
        for (float s : signal) {
            sumSq += s * s;
        }
        return (float) Math.sqrt(sumSq / signal.length);
    }

    /**
     * Wyznaczenie spektrogramu Log-Mel w skali decybelowej (dB).
     * Dokładnie odpowiada wywołaniu librosa.power_to_db(mel, ref=np.max, top_db=80.0).
     */
    private float[][] computeLogMelSpectrogram(float[] chunk) {
        float[][] power = stftTransformer.computePowerSpectrogram(chunk);
        float[][] mel = melFilterBank.apply(power);

        int numMels = mel.length;
        int timeSteps = mel[0].length;

        // Wyszukanie wartości maksymalnej jako referencji (ref = np.max)
        float maxVal = 0.0f;
        for (int m = 0; m < numMels; m++) {
            for (int t = 0; t < timeSteps; t++) {
                if (mel[m][t] > maxVal) {
                    maxVal = mel[m][t];
                }
            }
        }
        float ref = Math.max(maxVal, 1e-10f);

        // Konwersja do dB: 10 * log10(max(val, 1e-10) / ref) z odcięciem na poziomie -80 dB
        float[][] logMel = new float[numMels][timeSteps];
        for (int m = 0; m < numMels; m++) {
            for (int t = 0; t < timeSteps; t++) {
                float val = Math.max(mel[m][t], 1e-10f);
                float db = 10.0f * (float) Math.log10(val / ref);
                if (db < -DspConstants.TOP_DB) {
                    db = -DspConstants.TOP_DB;
                }
                logMel[m][t] = db;
            }
        }

        return logMel;
    }

    /**
     * Obliczenie kanału Delta (1. pochodna czasowa) z oknem Savitzky-Golay N = 9.
     * Odpowiada wywołaniu librosa.feature.delta(data, order=1, width=9).
     */
    private static float[][] computeDelta1(float[][] data) {
        int numMels = data.length;
        int timeSteps = data[0].length;
        float[][] delta = new float[numMels][timeSteps];

        float denom = 60.0f; // 2 * (1^2 + 2^2 + 3^2 + 4^2) = 60

        for (int m = 0; m < numMels; m++) {
            float[] row = data[m];
            float[] dRow = delta[m];

            // Punkty wewnętrzne (od 4 do timeSteps - 5)
            for (int t = 4; t < timeSteps - 4; t++) {
                float sum = 0.0f;
                for (int k = 1; k <= 4; k++) {
                    sum += k * (row[t + k] - row[t - k]);
                }
                dRow[t] = sum / denom;
            }

            // Warunki brzegowe zgodne z scipy savgol_filter mode='interp'
            float leftVal = dRow[4];
            float rightVal = dRow[timeSteps - 5];

            for (int t = 0; t < 4; t++) {
                dRow[t] = leftVal;
            }
            for (int t = timeSteps - 4; t < timeSteps; t++) {
                dRow[t] = rightVal;
            }
        }

        return delta;
    }

    /**
     * Obliczenie kanału Delta-Delta (2. pochodna czasowa) z oknem Savitzky-Golay N = 9.
     * Odpowiada wywołaniu librosa.feature.delta(data, order=2, width=9).
     */
    private static float[][] computeDelta2(float[][] data) {
        int numMels = data.length;
        int timeSteps = data[0].length;
        float[][] delta2 = new float[numMels][timeSteps];

        for (int m = 0; m < numMels; m++) {
            float[] row = data[m];
            float[] dRow = delta2[m];

            // Punkty wewnętrzne
            for (int t = 4; t < timeSteps - 4; t++) {
                float sum = 0.0f;
                for (int k = -4; k <= 4; k++) {
                    sum += SAVGOL_C2_KERNEL[k + 4] * row[t + k];
                }
                dRow[t] = sum;
            }

            // Warunki brzegowe
            float leftVal = dRow[4];
            float rightVal = dRow[timeSteps - 5];

            for (int t = 0; t < 4; t++) {
                dRow[t] = leftVal;
            }
            for (int t = timeSteps - 4; t < timeSteps; t++) {
                dRow[t] = rightVal;
            }
        }

        return delta2;
    }

    /**
     * Standaryzacja Z-score dla dwuwymiarowej macierzy kanału:
     * (x - mean) / std z zabezpieczeniem epsilon.
     */
    public static void normalizeChannelZScore(float[][] channel) {
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
        if (std < DspConstants.EPSILON) {
            std = DspConstants.EPSILON;
        }

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                channel[r][c] = (channel[r][c] - mean) / std;
            }
        }
    }
}
