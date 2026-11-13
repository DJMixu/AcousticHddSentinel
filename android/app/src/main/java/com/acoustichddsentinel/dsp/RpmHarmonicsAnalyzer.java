package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Analizator widmowy pików częstotliwościowych i grzebienia harmonicznego prędkości obrotowej (RPM) dysków HDD.
 * Zgodny z fizyką akustyczną HDD oraz niezmiennikami ADR-001.
 *
 * Wykrywa:
 * 1. Częstotliwości podstawowe i harmoniczne obrotu silnika wrzeciona:
 *    - 5400 RPM: f0 = 90.0 Hz (harmoniczne 180, 270, 360, 450, 540, 1080 Hz)
 *    - 7200 RPM: f0 = 120.0 Hz (harmoniczne 240, 360, 480, 600, 720, 1440 Hz)
 * 2. Brak rotacji i pisk stojana zablokowanego silnika (Stall Beep w paśmie 800 - 3500 Hz)
 * 3. Odrzucanie szumu wentylatorów PC i mowy ludzkiej
 */
public class RpmHarmonicsAnalyzer {

    private static final int FFT_SIZE = 4096;
    private static final float FREQ_BIN_WIDTH = (float) DspConstants.SAMPLE_RATE / FFT_SIZE; // ~5.38 Hz

    public static class RpmResult {
        public final boolean hasRotation;
        public final String detectedRpm;      // "5400_rpm", "7200_rpm" lub null
        public final int nominalRpm;          // 5400, 7200 lub 0
        public final float rpmConfidence;     // 0.0 - 1.0
        public final float f0FoundHz;
        public final boolean hasStallBeep;
        public final float stallBeepFreqHz;
        public final float rms;

        public RpmResult(boolean hasRotation, String detectedRpm, int nominalRpm,
                         float rpmConfidence, float f0FoundHz, boolean hasStallBeep,
                         float stallBeepFreqHz, float rms) {
            this.hasRotation = hasRotation;
            this.detectedRpm = detectedRpm;
            this.nominalRpm = nominalRpm;
            this.rpmConfidence = rpmConfidence;
            this.f0FoundHz = f0FoundHz;
            this.hasStallBeep = hasStallBeep;
            this.stallBeepFreqHz = stallBeepFreqHz;
            this.rms = rms;
        }

        public String getSummaryDescription() {
            if (hasRotation) {
                return String.format(java.util.Locale.US,
                    "Wykryto prawidłową rotację silnika HDD (%d RPM, f0=%.1f Hz).", nominalRpm, f0FoundHz);
            }
            if (hasStallBeep) {
                return String.format(java.util.Locale.US,
                    "Wykryto pisk stojana zatartego silnika wrzeciona (%.1f Hz). Brak obrotów!", stallBeepFreqHz);
            }
            return "Brak charakterystycznych sygnatur obrotowych dysku HDD (tło komputera / wentylator).";
        }
    }

    /**
     * Analizuje 3-sekundowy bufor próbek audio (66150 próbek przy 22050 Hz).
     */
    public RpmResult analyzeChunk(float[] samples) {
        if (samples == null || samples.length == 0) {
            return new RpmResult(false, null, 0, 0f, 0f, false, 0f, 0f);
        }

        // 1. Obliczenie RMS
        float sumSq = 0f;
        for (float s : samples) {
            sumSq += s * s;
        }
        float rms = (float) Math.sqrt(sumSq / samples.length);
        if (rms < DspConstants.RMS_SILENCE_THRESHOLD) {
            return new RpmResult(false, null, 0, 0f, 0f, false, 0f, rms);
        }

        // 2. Obliczenie uśrednionego widma Welch z oknem FFT_SIZE=4096 i nakładaniem 50%
        float[] avgSpectrum = computeWelchSpectrum(samples, FFT_SIZE);
        int numBins = avgSpectrum.length;

        // Średnia energia widmowa
        float totalEnergy = 0f;
        for (float v : avgSpectrum) {
            totalEnergy += v;
        }
        float meanEnergy = totalEnergy / Math.max(1, numBins);

        // 3. Ocena profilu 5400 RPM (f0 = 90 Hz) - rozszerzona o harmoniczne biegunów stojana 8x, 9x, 10x
        CombScore score5400 = evaluateComb(avgSpectrum, 90.0f, new float[]{180.0f, 270.0f, 360.0f, 450.0f, 540.0f, 720.0f, 810.0f, 900.0f, 1080.0f});

        // 4. Ocena profilu 7200 RPM (f0 = 120 Hz)
        CombScore score7200 = evaluateComb(avgSpectrum, 120.0f, new float[]{240.0f, 360.0f, 480.0f, 600.0f, 720.0f, 1440.0f});

        boolean hasRotation = false;
        String detectedRpm = null;
        int nominalRpm = 0;
        float confidence = 0f;
        float f0Found = 0f;

        if (score7200.detected && score7200.score >= score5400.score) {
            hasRotation = true;
            detectedRpm = "7200_rpm";
            nominalRpm = 7200;
            confidence = Math.min(1.0f, score7200.score / 10.0f);
            f0Found = score7200.exactF0Hz;
        } else if (score5400.detected) {
            hasRotation = true;
            detectedRpm = "5400_rpm";
            nominalRpm = 5400;
            confidence = Math.min(1.0f, score5400.score / 10.0f);
            f0Found = score5400.exactF0Hz;
        }

        // 5. Test pisku zatartego silnika (Stall Beep w paśmie 1050 - 3500 Hz)
        boolean hasStallBeep = false;
        float stallBeepFreq = 0f;

        if (!hasRotation) {
            int binMin = (int) (1050.0f / FREQ_BIN_WIDTH);
            int binMax = Math.min(numBins - 1, (int) (3500.0f / FREQ_BIN_WIDTH));
            float maxPeakVal = 0f;
            int maxPeakBin = -1;

            for (int b = binMin; b <= binMax; b++) {
                float val = avgSpectrum[b];
                if (val > maxPeakVal && val > meanEnergy * 6.0f) {
                    // Sprawdzenie lokalnej dominacji (szczyt lokalny)
                    if (b > 0 && b < numBins - 1 && val > avgSpectrum[b - 1] && val > avgSpectrum[b + 1]) {
                        maxPeakVal = val;
                        maxPeakBin = b;
                    }
                }
            }

            if (maxPeakBin > 0) {
                hasStallBeep = true;
                stallBeepFreq = maxPeakBin * FREQ_BIN_WIDTH;
            }
        }

        return new RpmResult(hasRotation, detectedRpm, nominalRpm, confidence, f0Found, hasStallBeep, stallBeepFreq, rms);
    }

    private static class CombScore {
        final boolean detected;
        final float score;
        final float exactF0Hz;

        CombScore(boolean detected, float score, float exactF0Hz) {
            this.detected = detected;
            this.score = score;
            this.exactF0Hz = exactF0Hz;
        }
    }

    private CombScore evaluateComb(float[] spectrum, float targetF0, float[] harmonics) {
        float f0Tolerance = 4.0f;
        int binF0Min = Math.max(0, (int) ((targetF0 - f0Tolerance) / FREQ_BIN_WIDTH));
        int binF0Max = Math.min(spectrum.length - 1, (int) ((targetF0 + f0Tolerance) / FREQ_BIN_WIDTH));

        int bestF0Bin = binF0Min;
        float maxF0Val = 0f;
        for (int b = binF0Min; b <= binF0Max; b++) {
            if (spectrum[b] > maxF0Val) {
                maxF0Val = spectrum[b];
                bestF0Bin = b;
            }
        }

        float exactF0Hz = bestF0Bin * FREQ_BIN_WIDTH;

        // Szacowanie lokalnego tła wokół f0 (poza pasmem piku)
        int bgBinMin = Math.max(0, (int) ((targetF0 - 20.0f) / FREQ_BIN_WIDTH));
        int bgBinMax = Math.min(spectrum.length - 1, (int) ((targetF0 + 20.0f) / FREQ_BIN_WIDTH));
        float bgSum = 0f;
        int bgCount = 0;
        for (int b = bgBinMin; b <= bgBinMax; b++) {
            if (b < binF0Min || b > binF0Max) {
                bgSum += spectrum[b];
                bgCount++;
            }
        }
        float localBg = bgCount > 0 ? (bgSum / bgCount) : 1e-6f;
        float f0Prominence = maxF0Val / Math.max(1e-6f, localBg);

        // Dopasowanie wyższych harmonicznych (tolerancja adaptacyjna proporcjonalna do częstotliwości)
        int harmonicHits = 0;
        for (float hFreq : harmonics) {
            float tol = Math.max(6.0f, hFreq * 0.02f);
            int hBinMin = Math.max(0, (int) ((hFreq - tol) / FREQ_BIN_WIDTH));
            int hBinMax = Math.min(spectrum.length - 1, (int) ((hFreq + tol) / FREQ_BIN_WIDTH));
            float maxHVal = 0f;
            for (int b = hBinMin; b <= hBinMax; b++) {
                if (spectrum[b] > maxHVal) {
                    maxHVal = spectrum[b];
                }
            }

            int hBgMin = Math.max(0, (int) ((hFreq - 25.0f) / FREQ_BIN_WIDTH));
            int hBgMax = Math.min(spectrum.length - 1, (int) ((hFreq + 25.0f) / FREQ_BIN_WIDTH));
            float hBgSum = 0f;
            int hBgCount = 0;
            for (int b = hBgMin; b <= hBgMax; b++) {
                if (b < hBinMin || b > hBinMax) {
                    hBgSum += spectrum[b];
                    hBgCount++;
                }
            }
            float hBg = hBgCount > 0 ? (hBgSum / hBgCount) : 1e-6f;
            if (maxHVal / Math.max(1e-6f, hBg) > 1.6f) {
                harmonicHits++;
            }
        }

        float score = (f0Prominence * 0.4f) + ((float) harmonicHits / harmonics.length * 6.0f);
        boolean detected = (f0Prominence >= 1.6f && harmonicHits >= 2) || (f0Prominence >= 3.0f && harmonicHits >= 1) || (harmonicHits >= 3);

        return new CombScore(detected, score, exactF0Hz);
    }

    /**
     * Oblicza uśrednione widmo amplitudowe z oknem Hann.
     */
    private float[] computeWelchSpectrum(float[] samples, int nFft) {
        int halfFft = nFft / 2 + 1;
        float[] avgSpec = new float[halfFft];
        int hop = nFft / 2;
        int numSegments = 0;

        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            window[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (nFft - 1))));
        }

        float[] real = new float[nFft];
        float[] imag = new float[nFft];

        for (int offset = 0; offset + nFft <= samples.length; offset += hop) {
            for (int i = 0; i < nFft; i++) {
                real[i] = samples[offset + i] * window[i];
                imag[i] = 0f;
            }

            radix2Fft(real, imag, nFft);

            for (int k = 0; k < halfFft; k++) {
                float mag = (float) Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
                avgSpec[k] += mag;
            }
            numSegments++;
        }

        if (numSegments > 0) {
            for (int k = 0; k < halfFft; k++) {
                avgSpec[k] /= numSegments;
            }
        }
        return avgSpec;
    }

    private void radix2Fft(float[] real, float[] imag, int n) {
        int j = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i < j) {
                float tr = real[i]; real[i] = real[j]; real[j] = tr;
                float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
            }
            int k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }

        for (int len = 2; len <= n; len <<= 1) {
            double angle = -2.0 * Math.PI / len;
            float wlen_r = (float) Math.cos(angle);
            float wlen_i = (float) Math.sin(angle);

            for (int i = 0; i < n; i += len) {
                float w_r = 1.0f;
                float w_i = 0.0f;
                int half = len / 2;

                for (int m = 0; m < half; m++) {
                    int u = i + m;
                    int v = i + m + half;

                    float vr = real[v] * w_r - imag[v] * w_i;
                    float vi = real[v] * w_i + imag[v] * w_r;

                    real[v] = real[u] - vr;
                    imag[v] = imag[u] - vi;
                    real[u] = real[u] + vr;
                    imag[u] = imag[u] + vi;

                    float next_w_r = w_r * wlen_r - w_i * wlen_i;
                    float next_w_i = w_r * wlen_i + w_i * wlen_r;
                    w_r = next_w_r;
                    w_i = next_w_i;
                }
            }
        }
    }
}
