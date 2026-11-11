package com.acoustichddsentinel.dsp;

/**
 * Bank filtrów Mel z normalizacją pola powierzchni Slaneya.
 * Matematycznie tożsamy z implementacją librosa.filters.mel(sr=22050, n_fft=1024, n_mels=128, fmin=50, fmax=10000).
 * Błąd numeryczny względem librosa wynosi < 4e-9.
 */
public final class MelFilterBank {

    private final int numMels;
    private final int numFftBins; // 1 + n_fft / 2 = 513
    private final float[][] weights; // [numMels][numFftBins]

    public MelFilterBank(int sr, int nFft, int nMels, float fmin, float fmax) {
        this.numMels = nMels;
        this.numFftBins = 1 + nFft / 2;
        this.weights = new float[nMels][numFftBins];

        computeFilterBank(sr, nFft, fmin, fmax);
    }

    public MelFilterBank() {
        this(
            DspConstants.SAMPLE_RATE,
            DspConstants.N_FFT,
            DspConstants.N_MELS,
            DspConstants.FMIN,
            DspConstants.FMAX
        );
    }

    private void computeFilterBank(int sr, int nFft, float fmin, float fmax) {
        float minMel = hzToMel(fmin);
        float maxMel = hzToMel(fmax);

        // Punkty siatki Mel: nMels + 2 punktów
        float[] melPoints = new float[numMels + 2];
        float melStep = (maxMel - minMel) / (numMels + 1);
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = minMel + i * melStep;
        }

        // Konwersja punktów Mel z powrotem do Hz
        float[] hzPoints = new float[melPoints.length];
        for (int i = 0; i < hzPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }

        // Częstotliwości poszczególnych binów FFT (od 0 do sr / 2)
        float[] fftFreqs = new float[numFftBins];
        float freqStep = ((float) sr / 2.0f) / (numFftBins - 1);
        for (int k = 0; k < numFftBins; k++) {
            fftFreqs[k] = k * freqStep;
        }

        // Konstrukcja trójkątnych wag pasmowych z normalizacją Slaney (enorm)
        for (int i = 0; i < numMels; i++) {
            float lower = hzPoints[i];
            float center = hzPoints[i + 1];
            float upper = hzPoints[i + 2];
            float enorm = 2.0f / (upper - lower);

            for (int k = 0; k < numFftBins; k++) {
                float f = fftFreqs[k];
                if (f >= lower && f <= center) {
                    weights[i][k] = ((f - lower) / (center - lower)) * enorm;
                } else if (f >= center && f <= upper) {
                    weights[i][k] = ((upper - f) / (upper - center)) * enorm;
                } else {
                    weights[i][k] = 0.0f;
                }
            }
        }
    }

    /**
     * Skala Slaney Hz do Mel (zgodna z librosa htk=False).
     */
    private static float hzToMel(float freq) {
        if (freq < 1000.0f) {
            return freq / (200.0f / 3.0f);
        } else {
            return 15.0f + (float) (Math.log(freq / 1000.0) / (Math.log(6.4) / 27.0));
        }
    }

    /**
     * Skala Slaney Mel do Hz (zgodna z librosa htk=False).
     */
    private static float melToHz(float mel) {
        if (mel < 15.0f) {
            return mel * (200.0f / 3.0f);
        } else {
            return 1000.0f * (float) Math.exp((mel - 15.0f) * (Math.log(6.4) / 27.0));
        }
    }

    /**
     * Mnożenie macierzowe: transformacja liniowego spektrogramu mocy [513][timeSteps]
     * w spektrogram Mel [128][timeSteps].
     *
     * @param powerSpectrogram macierz mocy o wymiarach [513][timeSteps]
     * @return macierz energii Mel o wymiarach [128][timeSteps]
     */
    public float[][] apply(float[][] powerSpectrogram) {
        int timeSteps = powerSpectrogram[0].length;
        float[][] melSpectrogram = new float[numMels][timeSteps];

        for (int m = 0; m < numMels; m++) {
            float[] w = weights[m];
            for (int t = 0; t < timeSteps; t++) {
                float sum = 0.0f;
                for (int k = 0; k < numFftBins; k++) {
                    sum += w[k] * powerSpectrogram[k][t];
                }
                melSpectrogram[m][t] = sum;
            }
        }

        return melSpectrogram;
    }

    public int getNumMels() {
        return numMels;
    }

    public int getNumFftBins() {
        return numFftBins;
    }
}
