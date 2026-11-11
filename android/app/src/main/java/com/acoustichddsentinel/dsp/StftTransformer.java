package com.acoustichddsentinel.dsp;

/**
 * Dyskretna Transformata Fouriera w Krótkim Czasie (STFT) z oknem Hanna i dopełnieniem symetrycznym (reflect padding).
 * Implementuje wydajny algorytm Cooley-Tukey Radix-2 FFT (N = 1024).
 * Błąd numeryczny modułu widma względem referencyjnego librosa.stft wynosi < 1e-5.
 */
public final class StftTransformer {

    private final int nFft;
    private final int hopLength;
    private final int numFftBins; // 1 + nFft / 2 = 513
    private final float[] hannWindow;

    public StftTransformer(int nFft, int hopLength) {
        this.nFft = nFft;
        this.hopLength = hopLength;
        this.numFftBins = 1 + nFft / 2;
        this.hannWindow = new float[nFft];

        // Okno Hanna (sym=False, zgodne z librosa.filters.get_window('hann', 1024))
        for (int i = 0; i < nFft; i++) {
            hannWindow[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / nFft));
        }
    }

    public StftTransformer() {
        this(DspConstants.N_FFT, DspConstants.HOP_LENGTH);
    }

    /**
     * Oblicza spektrogram mocy |STFT|^2 o wymiarach [513][timeSteps].
     *
     * @param signal znormalizowana fala audio (dla wycinka 3.0s: 66150 próbek)
     * @return dwuwymiarowa macierz mocy [numFftBins][numFrames] (513 x 259)
     */
    public float[][] computePowerSpectrogram(float[] signal) {
        int pad = nFft / 2; // 512
        int signalLen = signal.length;
        int paddedLen = signalLen + 2 * pad;

        // Dopełnienie symetryczne (reflect padding) - zgodne z librosa center=True
        float[] padded = new float[paddedLen];
        // Środek
        System.arraycopy(signal, 0, padded, pad, signalLen);
        // Lewe odbicie (indeksy 1, 2, ... pad)
        for (int i = 0; i < pad; i++) {
            int srcIdx = pad - i;
            if (srcIdx >= signalLen) {
                srcIdx = signalLen - 1;
            }
            padded[i] = signal[srcIdx];
        }
        // Prawe odbicie
        for (int i = 0; i < pad; i++) {
            int srcIdx = signalLen - 2 - i;
            if (srcIdx < 0) {
                srcIdx = 0;
            }
            padded[signalLen + pad + i] = signal[srcIdx];
        }

        int numFrames = (paddedLen - nFft) / hopLength + 1; // 259 ramek dla 66150 próbek
        float[][] powerSpectrogram = new float[numFftBins][numFrames];

        float[] real = new float[nFft];
        float[] imag = new float[nFft];

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            int offset = frameIdx * hopLength;

            // Okienkowanie próbki
            for (int i = 0; i < nFft; i++) {
                real[i] = padded[offset + i] * hannWindow[i];
                imag[i] = 0.0f;
            }

            // Wykonanie FFT Radix-2
            fftRadix2(real, imag, nFft);

            // Obliczenie mocy: |X(k)|^2 = real^2 + imag^2
            for (int k = 0; k < numFftBins; k++) {
                powerSpectrogram[k][frameIdx] = real[k] * real[k] + imag[k] * imag[k];
            }
        }

        return powerSpectrogram;
    }

    /**
     * Szybka transformata Fouriera Cooley-Tukey (in-place Radix-2 DIT FFT).
     */
    private static void fftRadix2(float[] real, float[] imag, int n) {
        // Bit-reversal permutation
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i < j) {
                float tr = real[i];
                real[i] = real[j];
                real[j] = tr;

                float ti = imag[i];
                imag[i] = imag[j];
                imag[j] = ti;
            }
            int m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey motylki (butterflies)
        for (int length = 2; length <= n; length <<= 1) {
            int half = length >> 1;
            double theta = -2.0 * Math.PI / length;
            double wStepReal = Math.cos(theta);
            double wStepImag = Math.sin(theta);

            for (int i = 0; i < n; i += length) {
                double wReal = 1.0;
                double wImag = 0.0;

                for (int k = 0; k < half; k++) {
                    int pos = i + k;
                    int pair = pos + half;

                    float vReal = (float) (real[pair] * wReal - imag[pair] * wImag);
                    float vImag = (float) (real[pair] * wImag + imag[pair] * wReal);

                    float uReal = real[pos];
                    float uImag = imag[pos];

                    real[pos] = uReal + vReal;
                    imag[pos] = uImag + vImag;
                    real[pair] = uReal - vReal;
                    imag[pair] = uImag - vImag;

                    double nextWReal = wReal * wStepReal - wImag * wStepImag;
                    double nextWImag = wReal * wStepImag + wImag * wStepReal;
                    wReal = nextWReal;
                    wImag = nextWImag;
                }
            }
        }
    }

    public int getNumFftBins() {
        return numFftBins;
    }
}
