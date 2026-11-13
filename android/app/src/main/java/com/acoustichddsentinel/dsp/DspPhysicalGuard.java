package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Fizyczny Strażnik DSP (Digital Signal Processing Physical Guard).
 * Port z modułu badawczego ml/dsp_adversarial_guard.py zgodny z ADR-001.
 *
 * Odpowiada za twarde weta fizyczne eliminujące fałszywe alarmy:
 * 1. Oczyszczacze powietrza i wentylatory PC (gładkie widmo 1/f, niska dynamika, brak transjentów)
 *    -> Weto dla 'clicking' (indeks 1)
 * 2. Przesterowania mikrofonu / clipping szczytów (hard clipping, niska płaskość widma >4kHz)
 *    -> Weto dla 'scratching' (indeks 3)
 * 3. Brak wąskopasmowego piku powyżej 1050 Hz lub potwierdzona rotacja wrzeciona
 *    -> Weto dla 'beeping' (indeks 0)
 */
public class DspPhysicalGuard {

    private static final int SR = DspConstants.SAMPLE_RATE; // 22050 Hz
    private static final float EPSILON = 1e-12f;

    public static class SanitizationResult {
        public final float[] sanitizedProbs;
        public final float crestFactor;
        public final int transientPeakCount;
        public final boolean isAerodynamicFanOrPurifier;
        public final boolean isMicrophoneClipped;
        public final boolean vetoClicking;
        public final boolean vetoScratching;
        public final boolean vetoBeeping;
        public final String auditLog;

        public SanitizationResult(float[] sanitizedProbs, float crestFactor, int transientPeakCount,
                                  boolean isAerodynamicFanOrPurifier, boolean isMicrophoneClipped,
                                  boolean vetoClicking, boolean vetoScratching, boolean vetoBeeping,
                                  String auditLog) {
            this.sanitizedProbs = sanitizedProbs;
            this.crestFactor = crestFactor;
            this.transientPeakCount = transientPeakCount;
            this.isAerodynamicFanOrPurifier = isAerodynamicFanOrPurifier;
            this.isMicrophoneClipped = isMicrophoneClipped;
            this.vetoClicking = vetoClicking;
            this.vetoScratching = vetoScratching;
            this.vetoBeeping = vetoBeeping;
            this.auditLog = auditLog;
        }
    }

    /**
     * Główna metoda sanityzacji prawdopodobieństw wyjściowych modelu.
     *
     * @param rawProbs surowe prawdopodobieństwa z Softmax [beeping, clicking, normal, scratching]
     * @param rawSamples próbki fali audio (wycinek 3.0s, 66150 próbek)
     * @param zNormLogMel macierz Log-Mel po Z-Score [128][259]
     * @param rpmRes wynik analizy harmonicznej RPM z RpmHarmonicsAnalyzer
     * @return wynik sanityzacji z przefiltrowanymi prawdopodobieństwami i logiem audytowym
     */
    public SanitizationResult sanitizeProbabilities(
            float[] rawProbs,
            float[] rawSamples,
            float[][] zNormLogMel,
            RpmHarmonicsAnalyzer.RpmResult rpmRes
    ) {
        if (rawProbs == null || rawProbs.length < 4) {
            return new SanitizationResult(rawProbs, 0f, 0, false, false, false, false, false, "Brak danych");
        }

        // 1. Usunięcie DC offsetu z sygnału fali
        float[] samples = removeDcOffset(rawSamples);

        // 2. Obliczenie cech fizycznych
        float rms = computeRms(samples);
        float peak = computePeak(samples);
        float crestFactor = peak / Math.max(rms, EPSILON);

        int transientCount = countTransientPeaks(samples, SR);
        boolean isFanOrPurifier = isAerodynamicFanOrPurifier(samples, crestFactor, transientCount);
        boolean isClipped = isMicrophoneClipped(samples, peak, crestFactor);

        float[] probs = Arrays.copyOf(rawProbs, rawProbs.length);
        boolean vetoClicking = false;
        boolean vetoScratching = false;
        boolean vetoBeeping = false;
        StringBuilder audit = new StringBuilder();

        // Indeksy klas: 0: beeping, 1: clicking, 2: normal, 3: scratching
        // WETO 1: CLICKING (indeks 1)
        // Autentyczny Click of Death wymaga wyrazistych uderzeń (transientCount >= 2) oraz wysokiego Crest Factor (>= 3.5).
        // Jednostajny szum oczyszczacza lub wentylatora nie może generować werdyktu clicking!
        if (crestFactor < 3.5f || transientCount < 2 || isFanOrPurifier) {
            if (probs[1] > 0.05f) {
                vetoClicking = true;
                float confiscated = probs[1] - 0.01f;
                probs[1] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto CLICKING: CF=%.2f, transients=%d, fan=%b] ",
                        crestFactor, transientCount, isFanOrPurifier));
            }
        }

        // WETO 2: SCRATCHING (indeks 3)
        // Fałszywe tarcie głowic powstaje przy przesterowaniu mikrofonu (clipping) lub szumie powietrza o niskim udziale >4kHz.
        float highFreqFraction = computeHighFrequencyEnergyFraction(samples, SR, 4000.0f);
        if (isClipped || (isFanOrPurifier && highFreqFraction < 0.15f)) {
            if (probs[3] > 0.05f) {
                vetoScratching = true;
                float confiscated = probs[3] - 0.01f;
                probs[3] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto SCRATCHING: clipped=%b, highFreq=%.1f%%] ",
                        isClipped, highFreqFraction * 100.0f));
            }
        }

        // WETO 3: BEEPING (indeks 0)
        // Dysk piszczy zablokowanym silnikiem wyłącznie, gdy silnik NIE wiruje i występuje pisk > 1050 Hz.
        boolean hasNormalRotation = (rpmRes != null && rpmRes.hasRotation);
        boolean hasStallTone = (rpmRes != null && rpmRes.hasStallBeep && rpmRes.stallBeepFreqHz >= 1050.0f);
        if (hasNormalRotation || !hasStallTone) {
            if (probs[0] > 0.05f) {
                vetoBeeping = true;
                float confiscated = probs[0] - 0.01f;
                probs[0] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto BEEPING: rot=%b, stallTone=%b] ",
                        hasNormalRotation, hasStallTone));
            }
        }

        // 3. Renormalizacja prawdopodobieństw do 1.0
        float sum = 0f;
        for (float p : probs) {
            sum += Math.max(p, 0f);
        }
        if (sum > EPSILON) {
            for (int i = 0; i < probs.length; i++) {
                probs[i] = Math.max(probs[i], 0f) / sum;
            }
        }

        return new SanitizationResult(
                probs,
                crestFactor,
                transientCount,
                isFanOrPurifier,
                isClipped,
                vetoClicking,
                vetoScratching,
                vetoBeeping,
                audit.toString().trim()
        );
    }

    /**
     * Usuwa składową stałą DC offset z sygnału.
     */
    public static float[] removeDcOffset(float[] raw) {
        if (raw == null || raw.length == 0) return new float[0];
        double sum = 0.0;
        for (float v : raw) {
            sum += v;
        }
        float mean = (float) (sum / raw.length);
        float[] out = new float[raw.length];
        for (int i = 0; i < raw.length; i++) {
            out[i] = raw[i] - mean;
        }
        return out;
    }

    private static float computeRms(float[] samples) {
        double sumSq = 0.0;
        for (float s : samples) {
            sumSq += s * s;
        }
        return (float) Math.sqrt(sumSq / Math.max(1, samples.length));
    }

    private static float computePeak(float[] samples) {
        float peak = 0f;
        for (float s : samples) {
            float abs = Math.abs(s);
            if (abs > peak) peak = abs;
        }
        return peak;
    }

    /**
     * Liczy wyraźne impulsy mechaniczne (transjenty) o separacji min. 80 ms,
     * przekraczające próg median + 2.2 * std w obwiedni krótkoczasowej.
     */
    public static int countTransientPeaks(float[] zeroMeanSamples, int sr) {
        int frameLen = 256; // ~11.6 ms
        int hop = 128;      // ~5.8 ms
        int nFrames = (zeroMeanSamples.length - frameLen) / hop + 1;
        if (nFrames < 4) return 0;

        float[] frameRms = new float[nFrames];
        for (int f = 0; f < nFrames; f++) {
            int offset = f * hop;
            double sumSq = 0.0;
            for (int i = 0; i < frameLen; i++) {
                float s = zeroMeanSamples[offset + i];
                sumSq += s * s;
            }
            frameRms[f] = (float) Math.sqrt(sumSq / frameLen);
        }

        // Obliczenie mediany i odchylenia standardowego energii
        float[] sorted = Arrays.copyOf(frameRms, frameRms.length);
        Arrays.sort(sorted);
        float median = sorted[sorted.length / 2];

        double sumDiffSq = 0.0;
        for (float v : frameRms) {
            float diff = v - median;
            sumDiffSq += diff * diff;
        }
        float std = (float) Math.sqrt(sumDiffSq / frameRms.length);
        float threshold = median + 2.2f * std;

        // Min separacja 80 ms w ramkach (80 ms / 5.8 ms ≈ 14 ramek)
        int minFrameDistance = (int) (0.080f * sr / hop);
        int transientCount = 0;
        int lastPeakFrame = -minFrameDistance;

        for (int f = 1; f < nFrames - 1; f++) {
            if (frameRms[f] > threshold && frameRms[f] > frameRms[f - 1] && frameRms[f] > frameRms[f + 1]) {
                if (f - lastPeakFrame >= minFrameDistance) {
                    transientCount++;
                    lastPeakFrame = f;
                }
            }
        }

        return transientCount;
    }

    /**
     * Sprawdza, czy sygnał to ciągły szum aerodynamiczny wentylatora lub oczyszczacza powietrza:
     * - Niski Crest Factor (< 4.5)
     * - Brak transjentów (< 2)
     * - Płynny spadek widma w paśmie 100 - 4000 Hz, mały udział energii > 4 kHz (< 15%).
     */
    public static boolean isAerodynamicFanOrPurifier(float[] zeroMeanSamples, float crestFactor, int transientCount) {
        if (crestFactor >= 4.5f || transientCount >= 2) {
            return false;
        }

        // Analiza pasmowa energii: pasmo niskie/średnie (100 - 1500 Hz) vs wysokie (> 4000 Hz)
        float lowMidEnergy = computeBandEnergy(zeroMeanSamples, SR, 100.0f, 1500.0f);
        float highEnergy = computeBandEnergy(zeroMeanSamples, SR, 4000.0f, 10000.0f);
        float totalEnergy = computeBandEnergy(zeroMeanSamples, SR, 50.0f, 10000.0f);

        if (totalEnergy < EPSILON) return false;

        float highRatio = highEnergy / totalEnergy;
        float lowMidRatio = lowMidEnergy / totalEnergy;

        // Szum turbulentny powietrza: dominuje pasmo <1.5 kHz (> 60%), pasmo >4 kHz znikome (< 12%)
        return (lowMidRatio > 0.60f && highRatio < 0.12f);
    }

    /**
     * Sprawdza, czy wystąpiło przesterowanie toru mikrofonu (Hard Clipping):
     * - Ponad 2% próbek osiąga >= 0.985 wartości szczytowej
     * - Lub występują spłaszczenia fali (flat run >= 3) przy niskim Crest Factor (< 2.5).
     */
    public static boolean isMicrophoneClipped(float[] zeroMeanSamples, float peak, float crestFactor) {
        if (peak < 0.01f || zeroMeanSamples.length == 0) return false;

        float railThresh = 0.985f * peak;
        int clippedSamples = 0;
        int maxFlatRun = 0;
        int currentFlatRun = 0;

        for (float s : zeroMeanSamples) {
            if (Math.abs(s) >= railThresh) {
                clippedSamples++;
                currentFlatRun++;
                if (currentFlatRun > maxFlatRun) {
                    maxFlatRun = currentFlatRun;
                }
            } else {
                currentFlatRun = 0;
            }
        }

        float clipRatio = (float) clippedSamples / zeroMeanSamples.length;
        return (clipRatio > 0.02f || (maxFlatRun >= 3 && crestFactor < 2.5f));
    }

    /**
     * Zwraca udział energii w paśmie powyżej danej częstotliwości granicznej.
     */
    private static float computeHighFrequencyEnergyFraction(float[] samples, int sr, float cutoffHz) {
        float highEnergy = computeBandEnergy(samples, sr, cutoffHz, 10000.0f);
        float totalEnergy = computeBandEnergy(samples, sr, 50.0f, 10000.0f);
        if (totalEnergy < EPSILON) return 0f;
        return highEnergy / totalEnergy;
    }

    /**
     * Oblicza przybliżoną energię w określonym paśmie częstotliwości za pomocą transformaty DFT.
     */
    private static float computeBandEnergy(float[] samples, int sr, float fLow, float fHigh) {
        int nFft = 2048;
        int step = nFft / 2;
        int nSteps = (samples.length - nFft) / step + 1;
        if (nSteps <= 0) return 0f;

        float binWidth = (float) sr / nFft;
        int binLow = Math.max(0, (int) (fLow / binWidth));
        int binHigh = Math.min(nFft / 2, (int) (fHigh / binWidth));

        double totalBandPower = 0.0;
        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            window[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / nFft));
        }

        for (int s = 0; s < nSteps; s++) {
            int offset = s * step;
            // DFT bezpośrednie tylko dla binów w wybranym paśmie
            for (int k = binLow; k <= binHigh; k++) {
                double real = 0.0;
                double imag = 0.0;
                double angleStep = 2.0 * Math.PI * k / nFft;
                for (int n = 0; n < nFft; n++) {
                    float val = samples[offset + n] * window[n];
                    double angle = angleStep * n;
                    real += val * Math.cos(angle);
                    imag -= val * Math.sin(angle);
                }
                totalBandPower += (real * real + imag * imag);
            }
        }

        return (float) (totalBandPower / (nSteps * nFft));
    }
}
