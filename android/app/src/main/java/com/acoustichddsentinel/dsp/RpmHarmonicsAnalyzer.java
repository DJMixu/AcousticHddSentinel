package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Utwardzony analizator widmowy pików częstotliwościowych i grzebienia harmonicznego RPM (Zero GC Churn).
 * Obsługuje 5400 / 7200 / 10000 / 15000 RPM, interpolację paraboliczną f0, ochronę przed siecią 50/60 Hz
 * oraz detekcję impulsowych pisków stojana (Max-Hold Stall Beep).
 */
public class RpmHarmonicsAnalyzer {

    private static final int FFT_SIZE = 4096;
    private static final float FREQ_BIN_WIDTH = (float) DspConstants.SAMPLE_RATE / FFT_SIZE; // ~5.3833 Hz

    private static final float[] MULTIPLIERS_5400 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 9.0f, 10.0f, 12.0f};
    private static final float[] MULTIPLIERS_7200 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 9.0f, 12.0f};
    private static final float[] MULTIPLIERS_10000 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 12.0f};
    private static final float[] MULTIPLIERS_15000 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    private static final float[] STATIONARITY_OFFSETS_FRAC = {0.0f, 0.37f, 0.83f, 1.41f, 2.11f};

    private final float[] hannWindow = new float[FFT_SIZE];
    private final float[] realBuffer = new float[FFT_SIZE];
    private final float[] imagBuffer = new float[FFT_SIZE];
    private final float[] avgPowerBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] maxPowerBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] avgSpectrumBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] maxSpectrumBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] medianScratchBuffer = new float[256];

    public RpmHarmonicsAnalyzer() {
        for (int i = 0; i < FFT_SIZE; i++) {
            hannWindow[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / FFT_SIZE)));
        }
    }

    public static class RpmResult {
        public final boolean hasRotation;
        public final String detectedRpm;
        public final int nominalRpm;
        public final float rpmConfidence;
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
                float expectedF0 = nominalRpm / 60.0f;
                float drift = f0FoundHz - expectedF0;
                if (Math.abs(drift) > 1.5f) {
                    return String.format(java.util.Locale.US,
                            "Wykryto niestabilną rotację wrzeciona HDD (%d RPM, f0=%.1f Hz, dryf=%.1f Hz - degradacja łożysk).",
                            nominalRpm, f0FoundHz, drift);
                }
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

    public synchronized RpmResult analyzeChunk(float[] samples) {
        if (samples == null || samples.length == 0) {
            return new RpmResult(false, null, 0, 0f, 0f, false, 0f, 0f);
        }

        // 1. Usunięcie DC offsetu (obliczenie średniej bez alokacji) i zero-mean RMS
        double sum = 0.0;
        for (float s : samples) {
            sum += s;
        }
        float dcMean = (float) (sum / samples.length);

        double sumSq = 0.0;
        for (float s : samples) {
            float d = s - dcMean;
            sumSq += d * d;
        }
        float rms = (float) Math.sqrt(sumSq / samples.length);
        if (rms < DspConstants.RMS_SILENCE_THRESHOLD) {
            return new RpmResult(false, null, 0, 0f, 0f, false, 0f, rms);
        }

        // 2. Obliczenie widma średniego Welcha oraz Max-Hold (Zero GC)
        computeWelchSpectra(samples, dcMean, FFT_SIZE);
        int numBins = avgSpectrumBuffer.length;

        float totalEnergy = 0f;
        for (float v : avgSpectrumBuffer) {
            totalEnergy += v;
        }
        float meanEnergy = totalEnergy / Math.max(1, numBins);

        // 3. Ewaluacja wszystkich 4 profili prędkości obrotowej
        CombScore[] candidates = new CombScore[]{
                evaluateCombDynamic(avgSpectrumBuffer, "5400_rpm", 5400, 90.0f, 4.5f, MULTIPLIERS_5400, meanEnergy),
                evaluateCombDynamic(avgSpectrumBuffer, "7200_rpm", 7200, 120.0f, 6.0f, MULTIPLIERS_7200, meanEnergy),
                evaluateCombDynamic(avgSpectrumBuffer, "10000_rpm", 10000, 166.67f, 6.0f, MULTIPLIERS_10000, meanEnergy),
                evaluateCombDynamic(avgSpectrumBuffer, "15000_rpm", 15000, 250.0f, 6.5f, MULTIPLIERS_15000, meanEnergy)
        };

        CombScore best = null;
        for (CombScore c : candidates) {
            if (c.detected && (best == null || c.score > best.score)) {
                best = c;
            }
        }

        boolean hasRotation = false;
        String detectedRpm = null;
        int nominalRpm = 0;
        float confidence = 0f;
        float f0Found = 0f;

        if (best != null && verifyTemporalPitchStationarity(samples, dcMean, best.exactF0Hz)) {
            hasRotation = true;
            detectedRpm = best.rpmKey;
            nominalRpm = best.nominalRpm;
            confidence = Math.min(1.0f, best.score / 10.0f);
            f0Found = best.exactF0Hz;
        }

        // 4. Test pisku zatartego silnika (Stall Beep 1050 - 3500 Hz)
        // Używa maxSpectrumBuffer, aby wychwycić także krótkie, cykliczne piski (np. 150 ms co 1.5 s)
        boolean hasStallBeep = false;
        float stallBeepFreq = 0f;

        if (!hasRotation) {
            stallBeepFreq = detectStallBeepFreq(avgSpectrumBuffer, meanEnergy, 3.0f);
            if (stallBeepFreq <= 0f) {
                float maxMean = 0f;
                for (float v : maxSpectrumBuffer) {
                    maxMean += v;
                }
                maxMean /= Math.max(1, numBins);
                // Dla Max-Hold wymagamy nieco wyższej ostrości (>= 4.0x) przeciwko szerokopasmowym klikom
                stallBeepFreq = detectStallBeepFreq(maxSpectrumBuffer, maxMean, 4.0f);
            }
            hasStallBeep = (stallBeepFreq > 0f);
        }

        return new RpmResult(hasRotation, detectedRpm, nominalRpm, confidence, f0Found, hasStallBeep, stallBeepFreq, rms);
    }

    private float detectStallBeepFreq(float[] spectrum, float meanEnergy, float minProminence) {
        int numBins = spectrum.length;
        int binMin = (int) (1050.0f / FREQ_BIN_WIDTH);
        int binMax = Math.min(numBins - 2, (int) (3500.0f / FREQ_BIN_WIDTH));

        double beepBandEnergy = 0.0;
        double fullSpectrumEnergy = 1e-12;
        for (int b = 0; b < numBins; b++) {
            double pwr = (double) spectrum[b] * spectrum[b];
            fullSpectrumEnergy += pwr;
            if (b >= binMin && b <= binMax) {
                beepBandEnergy += pwr;
            }
        }
        if (beepBandEnergy / fullSpectrumEnergy < 0.005) {
            return 0f;
        }

        float maxPeakVal = 0f;
        int maxPeakBin = -1;
        for (int b = Math.max(1, binMin); b <= binMax; b++) {
            float val = spectrum[b];
            if (val > maxPeakVal && val > meanEnergy * 5.0f && val > spectrum[b - 1] && val > spectrum[b + 1]) {
                int bgLow = Math.max(0, b - 12);
                int bgHigh = Math.min(numBins - 1, b + 12);
                int bgCount = 0;
                for (int k = bgLow; k <= bgHigh; k++) {
                    if (Math.abs(k - b) > 2 && bgCount < medianScratchBuffer.length) {
                        medianScratchBuffer[bgCount++] = spectrum[k];
                    }
                }
                float localBg = (bgCount > 0) ? computeMedian(medianScratchBuffer, bgCount) : meanEnergy;
                if (val / Math.max(localBg, 1e-12f) >= minProminence) {
                    maxPeakVal = val;
                    maxPeakBin = b;
                }
            }
        }

        if (maxPeakBin <= 0) {
            return 0f;
        }
        return interpolatePeakFreqHz(spectrum, maxPeakBin);
    }

    private static class CombScore {
        final String rpmKey;
        final int nominalRpm;
        final boolean detected;
        final float score;
        final float exactF0Hz;

        CombScore(String rpmKey, int nominalRpm, boolean detected, float score, float exactF0Hz) {
            this.rpmKey = rpmKey;
            this.nominalRpm = nominalRpm;
            this.detected = detected;
            this.score = score;
            this.exactF0Hz = exactF0Hz;
        }
    }

    private CombScore evaluateCombDynamic(
            float[] spectrum,
            String rpmKey,
            int nominalRpm,
            float targetF0,
            float f0Tolerance,
            float[] multipliers,
            float meanEnergy
    ) {
        int binF0Min = Math.max(1, (int) Math.floor((targetF0 - f0Tolerance) / FREQ_BIN_WIDTH));
        int binF0Max = Math.min(spectrum.length - 2, (int) Math.ceil((targetF0 + f0Tolerance) / FREQ_BIN_WIDTH));

        double totalEnergy = 1e-12;
        for (float v : spectrum) {
            totalEnergy += (double) v * v;
        }

        int bestF0Bin = binF0Min;
        float maxF0Val = 0f;
        boolean foundLocalPeak = false;
        for (int b = binF0Min; b <= binF0Max; b++) {
            float val = spectrum[b];
            boolean isPeak = (val >= spectrum[b - 1] && val >= spectrum[b + 1]);
            if (isPeak) {
                if (!foundLocalPeak || val > maxF0Val) {
                    foundLocalPeak = true;
                    maxF0Val = val;
                    bestF0Bin = b;
                }
            } else if (!foundLocalPeak && val > maxF0Val) {
                maxF0Val = val;
                bestF0Bin = b;
            }
        }

        // Interpolacja paraboliczna usuwa błąd kwantyzacji binu 5.38 Hz przed mnożeniem przez m <= 12
        float exactF0Hz = interpolatePeakFreqHz(spectrum, bestF0Bin);

        // Odrzucenie przydźwięku sieci energetycznej (60 Hz -> 120 Hz; 50/100 Hz -> 250 Hz)
        if (isMainsHumConfounder(spectrum, nominalRpm, maxF0Val)) {
            return new CombScore(rpmKey, nominalRpm, false, 0f, exactF0Hz);
        }

        double harmonicEnergy = (double) maxF0Val * maxF0Val;

        float localBg = computeLocalMedianBackground(spectrum, bestF0Bin, 25.0f, 3.5f);
        float f0Prominence = maxF0Val / Math.max(1e-6f, localBg);

        float f0Sharpness = 1.0f;
        if (bestF0Bin >= 2 && bestF0Bin + 2 < spectrum.length) {
            float skirt = (spectrum[bestF0Bin - 2] + spectrum[bestF0Bin + 2]) * 0.5f;
            f0Sharpness = maxF0Val / Math.max(skirt, 1e-9f);
        }

        int harmonicHits = 0;
        boolean has2ndHarmonicHit = false;
        for (float m : multipliers) {
            float hFreq = exactF0Hz * m;
            if (hFreq >= (DspConstants.SAMPLE_RATE / 2.0f) - 60.0f) {
                continue;
            }

            float tol = Math.max(6.0f, hFreq * 0.02f);
            int hBinMin = Math.max(0, (int) ((hFreq - tol) / FREQ_BIN_WIDTH));
            int hBinMax = Math.min(spectrum.length - 1, (int) ((hFreq + tol) / FREQ_BIN_WIDTH));

            float maxHVal = 0f;
            for (int b = hBinMin; b <= hBinMax; b++) {
                if (spectrum[b] > maxHVal) {
                    maxHVal = spectrum[b];
                }
            }

            float hBgWindow = Math.max(40.0f, tol * 2.0f);
            int hBgMin = Math.max(0, (int) ((hFreq - hBgWindow) / FREQ_BIN_WIDTH));
            int hBgMax = Math.min(spectrum.length - 1, (int) ((hFreq + hBgWindow) / FREQ_BIN_WIDTH));
            int hBgCount = 0;
            for (int b = hBgMin; b <= hBgMax; b++) {
                if ((b < hBinMin || b > hBinMax) && hBgCount < medianScratchBuffer.length) {
                    medianScratchBuffer[hBgCount++] = spectrum[b];
                }
            }
            float hBg = (hBgCount > 0) ? computeMedian(medianScratchBuffer, hBgCount) : 1e-6f;
            if (maxHVal / Math.max(1e-6f, hBg) > 1.6f && maxHVal > meanEnergy * 0.05f) {
                harmonicHits++;
                harmonicEnergy += (double) maxHVal * maxHVal;
                if (Math.abs(m - 2.0f) < 0.01f) {
                    has2ndHarmonicHit = true;
                }
            }
        }

        float score = (Math.min(f0Prominence, 10.0f) * 0.4f) + ((float) harmonicHits / multipliers.length * 6.0f);
        float energyRatio = (float) (harmonicEnergy / totalEnergy);

        boolean multiHitComb = harmonicHits >= 2
                && score >= 3.0f
                && ((f0Prominence >= 2.2f) || (f0Prominence >= 1.8f && harmonicHits >= 3 && score >= 3.8f));

        boolean single2ndHarmonicComb = harmonicHits >= 1
                && has2ndHarmonicHit
                && f0Prominence >= 3.0f
                && score >= 3.0f
                && f0Sharpness >= 6.0f;

        boolean detected = (multiHitComb || single2ndHarmonicComb)
                && (energyRatio >= 0.005f || harmonicHits >= 4);

        return new CombScore(rpmKey, nominalRpm, detected, score, exactF0Hz);
    }

    private boolean isMainsHumConfounder(float[] spectrum, int nominalRpm, float f0PeakVal) {
        if (nominalRpm == 7200) {
            int bin60 = Math.round(60.0f / FREQ_BIN_WIDTH);
            if (bin60 >= 1 && bin60 < spectrum.length - 1) {
                float bg60 = computeLocalMedianBackground(spectrum, bin60, 20.0f, 5.5f);
                if (spectrum[bin60] / Math.max(bg60, 1e-6f) >= 3.0f && spectrum[bin60] >= 0.75f * f0PeakVal) {
                    return true;
                }
            }
        } else if (nominalRpm == 15000) {
            int bin50 = Math.round(50.0f / FREQ_BIN_WIDTH);
            int bin100 = Math.round(100.0f / FREQ_BIN_WIDTH);
            if (bin50 >= 1 && bin100 < spectrum.length - 1) {
                float p50 = spectrum[bin50] / Math.max(computeLocalMedianBackground(spectrum, bin50, 20.0f, 5.5f), 1e-6f);
                float p100 = spectrum[bin100] / Math.max(computeLocalMedianBackground(spectrum, bin100, 20.0f, 5.5f), 1e-6f);
                if (p50 >= 3.0f && p100 >= 2.5f) {
                    return true;
                }
            }
        }
        return false;
    }

    private float computeLocalMedianBackground(float[] spectrum, int centerBin, float radiusHz, float excludeHz) {
        float centerFreq = centerBin * FREQ_BIN_WIDTH;
        int minBin = Math.max(0, (int) ((centerFreq - radiusHz) / FREQ_BIN_WIDTH));
        int maxBin = Math.min(spectrum.length - 1, (int) ((centerFreq + radiusHz) / FREQ_BIN_WIDTH));
        int count = 0;
        for (int b = minBin; b <= maxBin; b++) {
            if (Math.abs(b * FREQ_BIN_WIDTH - centerFreq) > excludeHz && count < medianScratchBuffer.length) {
                medianScratchBuffer[count++] = spectrum[b];
            }
        }
        return (count > 0) ? computeMedian(medianScratchBuffer, count) : 1e-6f;
    }

    private static float interpolatePeakFreqHz(float[] spectrum, int peakBin) {
        if (peakBin <= 0 || peakBin >= spectrum.length - 1) {
            return peakBin * FREQ_BIN_WIDTH;
        }
        float alpha = spectrum[peakBin - 1];
        float beta = spectrum[peakBin];
        float gamma = spectrum[peakBin + 1];
        float denom = 2.0f * (2.0f * beta - alpha - gamma);
        float delta = (Math.abs(denom) > 1e-12f) ? ((gamma - alpha) / denom) : 0.0f;
        delta = Math.max(-0.5f, Math.min(0.5f, delta));
        return (peakBin + delta) * FREQ_BIN_WIDTH;
    }

    private float computeMedian(float[] buf, int count) {
        if (count <= 0) return 1e-6f;
        if (count == 1) return buf[0];
        Arrays.sort(buf, 0, count);
        if ((count & 1) == 1) {
            return buf[count / 2];
        }
        return 0.5f * (buf[count / 2 - 1] + buf[count / 2]);
    }

    private void computeWelchSpectra(float[] samples, float dcMean, int nFft) {
        int halfFft = nFft / 2 + 1;
        Arrays.fill(avgPowerBuffer, 0f);
        Arrays.fill(maxPowerBuffer, 0f);
        Arrays.fill(avgSpectrumBuffer, 0f);
        Arrays.fill(maxSpectrumBuffer, 0f);

        int hop = nFft / 2;
        int numSegments = 0;

        for (int offset = 0; offset + nFft <= samples.length; offset += hop) {
            for (int i = 0; i < nFft; i++) {
                realBuffer[i] = (samples[offset + i] - dcMean) * hannWindow[i];
                imagBuffer[i] = 0f;
            }

            radix2Fft(realBuffer, imagBuffer, nFft);

            for (int k = 0; k < halfFft; k++) {
                float pwr = realBuffer[k] * realBuffer[k] + imagBuffer[k] * imagBuffer[k];
                avgPowerBuffer[k] += pwr;
                if (pwr > maxPowerBuffer[k]) {
                    maxPowerBuffer[k] = pwr;
                }
            }
            numSegments++;
        }

        if (numSegments > 0) {
            float invSeg = 1.0f / numSegments;
            for (int k = 0; k < halfFft; k++) {
                avgSpectrumBuffer[k] = (float) Math.sqrt(avgPowerBuffer[k] * invSeg);
                maxSpectrumBuffer[k] = (float) Math.sqrt(maxPowerBuffer[k]);
            }
        }
    }

    /**
     * Radix-2 DIT FFT z podwójną precyzją czynników obrotu (eliminacja dryfu fazy dla N=4096).
     */
    private static void radix2Fft(float[] real, float[] imag, int n) {
        int j = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i < j) {
                float tr = real[i]; real[i] = real[j]; real[j] = tr;
                float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
            }
            int k = n >> 1;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        for (int len = 2; len <= n; len <<= 1) {
            int half = len >> 1;
            double angle = -2.0 * Math.PI / len;
            double wStepR = Math.cos(angle);
            double wStepI = Math.sin(angle);

            for (int i = 0; i < n; i += len) {
                double wR = 1.0;
                double wI = 0.0;

                for (int m = 0; m < half; m++) {
                    int u = i + m;
                    int v = u + half;

                    float vr = (float) (real[v] * wR - imag[v] * wI);
                    float vi = (float) (real[v] * wI + imag[v] * wR);

                    float ur = real[u];
                    float ui = imag[u];

                    real[u] = ur + vr;
                    imag[u] = ui + vi;
                    real[v] = ur - vr;
                    imag[v] = ui - vi;

                    double nextWR = wR * wStepR - wI * wStepI;
                    double nextWI = wR * wStepI + wI * wStepR;
                    wR = nextWR;
                    wI = nextWI;
                }
            }
        }
    }

    private boolean verifyTemporalPitchStationarity(float[] samples, float dcMean, float f0Found) {
        if (samples.length < DspConstants.SAMPLE_RATE * 2.5f) {
            return true;
        }

        int nFft = FFT_SIZE;
        int kMin = Math.max(1, (int) Math.floor((f0Found - 5.0f) / FREQ_BIN_WIDTH));
        int kMax = Math.min(nFft / 2 - 2, (int) Math.ceil((f0Found + 5.0f) / FREQ_BIN_WIDTH));
        if (kMin > kMax) {
            return true;
        }

        float minFSub = Float.MAX_VALUE;
        float maxFSub = -Float.MAX_VALUE;
        int validCount = 0;

        for (float frac : STATIONARITY_OFFSETS_FRAC) {
            int offset = (int) (frac * DspConstants.SAMPLE_RATE);
            if (offset + nFft > samples.length) {
                continue;
            }

            for (int i = 0; i < nFft; i++) {
                realBuffer[i] = (samples[offset + i] - dcMean) * hannWindow[i];
                imagBuffer[i] = 0f;
            }

            radix2Fft(realBuffer, imagBuffer, nFft);

            int bestK = kMin;
            float maxVal = 0f;
            for (int k = kMin; k <= kMax; k++) {
                float mag = (float) Math.sqrt(realBuffer[k] * realBuffer[k] + imagBuffer[k] * imagBuffer[k]);
                if (mag > maxVal) {
                    maxVal = mag;
                    bestK = k;
                }
            }

            if (bestK > 0 && bestK < nFft / 2) {
                float alpha = (float) Math.sqrt(realBuffer[bestK - 1] * realBuffer[bestK - 1] + imagBuffer[bestK - 1] * imagBuffer[bestK - 1]);
                float beta = maxVal;
                float gamma = (float) Math.sqrt(realBuffer[bestK + 1] * realBuffer[bestK + 1] + imagBuffer[bestK + 1] * imagBuffer[bestK + 1]);
                float denom = 2.0f * (2.0f * beta - alpha - gamma);
                float delta = (Math.abs(denom) > 1e-12f) ? ((gamma - alpha) / denom) : 0.0f;
                delta = Math.max(-0.5f, Math.min(0.5f, delta));
                float fSub = (bestK + delta) * FREQ_BIN_WIDTH;

                if (fSub < minFSub) minFSub = fSub;
                if (fSub > maxFSub) maxFSub = fSub;
                validCount++;
            }
        }

        if (validCount < 3) {
            return true;
        }
        return (maxFSub - minFSub) <= 1.5f;
    }
}
