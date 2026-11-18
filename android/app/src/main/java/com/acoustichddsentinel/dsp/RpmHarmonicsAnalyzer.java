package com.acoustichddsentinel.dsp;

import java.util.Arrays;

/**
 * Analizator widmowy pików częstotliwościowych i grzebienia harmonicznego prędkości obrotowej (RPM) dysków HDD.
 * Zgodny z fizyką akustyczną HDD oraz niezmiennikami ADR-001.
 *
 * Wykrywa:
 * 1. Dynamiczne śledzenie częstotliwości podstawowej i harmonicznych obrotu silnika (Dynamic Harmonic Tracking):
 *    - 5400 RPM: nominalne f0 = 90.0 Hz z dynamicznym oknem śledzenia +/- 4.5 Hz (85.5 - 94.5 Hz)
 *      Harmoniczne stojana i biegunów: 2x, 3x, 4x, 5x, 6x, 8x, 9x, 10x, 12x
 *    - 7200 RPM: nominalne f0 = 120.0 Hz z dynamicznym oknem śledzenia +/- 6.0 Hz (114.0 - 126.0 Hz)
 *      Harmoniczne: 2x, 3x, 4x, 5x, 6x, 12x
 *    Obsługuje spadek prędkości obrotowej w agonalnych dyskach 2.5" z uszkodzonymi łożyskami.
 * 2. Brak rotacji i pisk stojana zablokowanego silnika (Stall Beep w paśmie 1050 - 3500 Hz).
 * 3. Odrzucanie szumu wentylatorów PC i basowego szumu otoczenia.
 *
 * Optymalizacje wydajnościowe (Java 17 / Android):
 * - Prealokacja buforów roboczych FFT, okna Hanna, widma mocy i mediany (Zero GC Allocation).
 * - Okno okresowe Hanna (fftbins=True, dzielenie przez nFft) zgodne z scipy.signal.welch.
 * - Uśrednianie widma mocy PSD z pierwiastkiem kwadratowym (scaling='spectrum').
 * - Obliczanie mediany lokalnego tła odporne na zakłócenia sąsiednich prążków.
 */
public class RpmHarmonicsAnalyzer {

    private static final int FFT_SIZE = 4096;
    private static final float FREQ_BIN_WIDTH = (float) DspConstants.SAMPLE_RATE / FFT_SIZE; // ~5.383 Hz

    // Mnożniki harmonicznych obrotu dla 5400 RPM (w tym harmoniczne biegunów stojana 8x, 9x, 10x)
    private static final float[] MULTIPLIERS_5400 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 9.0f, 10.0f, 12.0f};

    // Mnożniki harmonicznych obrotu dla 7200 RPM
    private static final float[] MULTIPLIERS_7200 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 12.0f};

    // Niespójne (incommensurate) przesunięcia czasowe do weryfikacji stacjonarności rotacji
    private static final float[] STATIONARITY_OFFSETS_FRAC = {0.0f, 0.37f, 0.83f, 1.41f, 2.11f};

    // Prealokowane bufory robocze - Zero GC Allocation podczas analizy widmowej
    private final float[] hannWindow = new float[FFT_SIZE];
    private final float[] realBuffer = new float[FFT_SIZE];
    private final float[] imagBuffer = new float[FFT_SIZE];
    private final float[] avgPowerBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] avgSpectrumBuffer = new float[FFT_SIZE / 2 + 1];
    private final float[] medianScratchBuffer = new float[256];

    public RpmHarmonicsAnalyzer() {
        // Okno okresowe Hanna (zgodne z scipy.signal.welch, sym=False)
        for (int i = 0; i < FFT_SIZE; i++) {
            hannWindow[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / FFT_SIZE)));
        }
    }

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
                float expectedF0 = nominalRpm == 5400 ? 90.0f : 120.0f;
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

    /**
     * Analizuje 3-sekundowy bufor próbek audio (66150 próbek przy 22050 Hz).
     */
    public synchronized RpmResult analyzeChunk(float[] samples) {
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

        // 3. Dynamiczne śledzenie harmoniczne dla 5400 RPM (zakres f0: 90.0 +/- 4.5 Hz)
        CombScore score5400 = evaluateCombDynamic(avgSpectrum, 90.0f, 4.5f, MULTIPLIERS_5400, meanEnergy);

        // 4. Dynamiczne śledzenie harmoniczne dla 7200 RPM (zakres f0: 120.0 +/- 6.0 Hz)
        CombScore score7200 = evaluateCombDynamic(avgSpectrum, 120.0f, 6.0f, MULTIPLIERS_7200, meanEnergy);

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

        // Weryfikacja stacjonarności rotacji wrzeciona (odrzucenie modulacji głosu i syren)
        if (hasRotation && !verifyTemporalPitchStationarity(samples, f0Found)) {
            hasRotation = false;
            detectedRpm = null;
            nominalRpm = 0;
            confidence = 0f;
            f0Found = 0f;
        }

        // 5. Test pisku zatartego silnika (Stall Beep w paśmie 1050 - 3500 Hz)
        boolean hasStallBeep = false;
        float stallBeepFreq = 0f;

        if (!hasRotation) {
            int binMin = (int) (1050.0f / FREQ_BIN_WIDTH);
            int binMax = Math.min(numBins - 1, (int) (3500.0f / FREQ_BIN_WIDTH));

            // Weryfikacja minimalnego udziału energii pasma pisku stojana (min. 0.5% całkowitej energii)
            double beepBandEnergy = 0.0;
            double fullSpectrumEnergy = 1e-12;
            for (int b = 0; b < numBins; b++) {
                float v = avgSpectrum[b];
                double pwr = (double) v * v;
                fullSpectrumEnergy += pwr;
                if (b >= binMin && b <= binMax) {
                    beepBandEnergy += pwr;
                }
            }

            if (beepBandEnergy / fullSpectrumEnergy >= 0.005) {
                float maxPeakVal = 0f;
                int maxPeakBin = -1;

                for (int b = binMin; b <= binMax; b++) {
                    float val = avgSpectrum[b];
                    if (val > maxPeakVal && val > meanEnergy * 5.0f) {
                        // Sprawdzenie lokalnej dominacji (szczyt lokalny)
                        if (b > 0 && b < numBins - 1 && val > avgSpectrum[b - 1] && val > avgSpectrum[b + 1]) {
                            // Weryfikacja wyrazistości piku (prominence) względem mediany lokalnego tła w otoczeniu +/-12 binów
                            int bgLow = Math.max(0, b - 12);
                            int bgHigh = Math.min(numBins - 1, b + 12);
                            int bgCount = 0;
                            for (int k = bgLow; k <= bgHigh; k++) {
                                if (Math.abs(k - b) > 2) {
                                    if (bgCount < medianScratchBuffer.length) {
                                        medianScratchBuffer[bgCount++] = avgSpectrum[k];
                                    }
                                }
                            }
                            float localBg = (bgCount > 0) ? computeMedian(medianScratchBuffer, bgCount) : meanEnergy;
                            float prominenceRatio = val / Math.max(localBg, 1e-12f);

                            // Autentyczny pisk stojana zablokowanego silnika to czysty, wąskopasmowy ton sinusoidalny (prominence >= 3.0x)
                            if (prominenceRatio >= 3.0f) {
                                maxPeakVal = val;
                                maxPeakBin = b;
                            }
                        }
                    }
                }

                if (maxPeakBin > 0) {
                    hasStallBeep = true;
                    stallBeepFreq = maxPeakBin * FREQ_BIN_WIDTH;
                }
            }
        }

        return new RpmResult(hasRotation, detectedRpm, nominalRpm, confidence, f0Found, hasStallBeep, stallBeepFreq, rms);
    }

    private static class CombScore {
        final boolean detected;
        final float score;
        final float exactF0Hz;
        final int harmonicHits;
        final float f0Prominence;

        CombScore(boolean detected, float score, float exactF0Hz, int harmonicHits, float f0Prominence) {
            this.detected = detected;
            this.score = score;
            this.exactF0Hz = exactF0Hz;
            this.harmonicHits = harmonicHits;
            this.f0Prominence = f0Prominence;
        }
    }

    /**
     * Dynamic Harmonic Tracking via Pitch Harmonizer.
     * Znajduje optymalną częstotliwość f0 w tolerancji f0Tolerance, a następnie dynamicznie
     * śledzi harmoniczne f_k = k * exactF0Hz, uwzględniając spadek prędkości obrotowej wywołany uszkodzeniem łożysk.
     */
    private CombScore evaluateCombDynamic(float[] spectrum, float targetF0, float f0Tolerance, float[] multipliers, float meanEnergy) {
        int binF0Min = Math.max(1, (int) Math.ceil((targetF0 - f0Tolerance) / FREQ_BIN_WIDTH));
        int binF0Max = Math.min(spectrum.length - 2, (int) Math.floor((targetF0 + f0Tolerance) / FREQ_BIN_WIDTH));

        double totalEnergy = 1e-12;
        for (float v : spectrum) {
            totalEnergy += v * v;
        }

        // 1. Wyszukanie optymalnego f0 w rozszerzonym oknie tolerancji (szczyt lokalny)
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

        float exactF0Hz = bestF0Bin * FREQ_BIN_WIDTH;
        double harmonicEnergy = (double) maxF0Val * maxF0Val;

        // 2. Szacowanie mediany lokalnego tła wokół targetF0 w promieniu +/- 25.0 Hz (poza pasmem piku)
        int bgBinMin = Math.max(0, (int) ((targetF0 - 25.0f) / FREQ_BIN_WIDTH));
        int bgBinMax = Math.min(spectrum.length - 1, (int) ((targetF0 + 25.0f) / FREQ_BIN_WIDTH));
        int bgCount = 0;
        for (int b = bgBinMin; b <= bgBinMax; b++) {
            float freq = b * FREQ_BIN_WIDTH;
            if (Math.abs(freq - exactF0Hz) > 3.5f) {
                if (bgCount < medianScratchBuffer.length) {
                    medianScratchBuffer[bgCount++] = spectrum[b];
                }
            }
        }
        float localBg = bgCount > 0 ? computeMedian(medianScratchBuffer, bgCount) : 1e-6f;
        float f0Prominence = maxF0Val / Math.max(1e-6f, localBg);

        // Wyznaczenie ostrości piku f0 względem ramion piku (+/- 2 biny widma)
        float f0Sharpness = 1.0f;
        if (bestF0Bin >= 2 && bestF0Bin + 2 < spectrum.length) {
            float skirt = (spectrum[bestF0Bin - 2] + spectrum[bestF0Bin + 2]) * 0.5f;
            f0Sharpness = maxF0Val / Math.max(skirt, 1e-9f);
        }

        // 3. Dynamiczne śledzenie wyższych harmonicznych proporcjonalnie do znalezionego exactF0Hz
        int harmonicHits = 0;
        boolean has2ndHarmonicHit = false;
        for (float m : multipliers) {
            float hFreq = exactF0Hz * m;
            if (hFreq >= (DspConstants.SAMPLE_RATE / 2.0f) - 60.0f) {
                continue;
            }

            // Adaptacyjna tolerancja zależna od rzędu harmonicznej
            float tol = Math.max(6.0f, hFreq * 0.025f);
            int hBinMin = Math.max(0, (int) ((hFreq - tol) / FREQ_BIN_WIDTH));
            int hBinMax = Math.min(spectrum.length - 1, (int) ((hFreq + tol) / FREQ_BIN_WIDTH));

            float maxHVal = 0f;
            for (int b = hBinMin; b <= hBinMax; b++) {
                if (spectrum[b] > maxHVal) {
                    maxHVal = spectrum[b];
                }
            }

            // Dynamiczny promień okna tła lokalnego dla harmonicznej (zapobiega wyzerowaniu binów dla hFreq > 1200 Hz)
            float hBgWindow = Math.max(40.0f, tol * 2.0f);
            int hBgMin = Math.max(0, (int) ((hFreq - hBgWindow) / FREQ_BIN_WIDTH));
            int hBgMax = Math.min(spectrum.length - 1, (int) ((hFreq + hBgWindow) / FREQ_BIN_WIDTH));
            int hBgCount = 0;
            for (int b = hBgMin; b <= hBgMax; b++) {
                if (b < hBinMin || b > hBinMax) {
                    if (hBgCount < medianScratchBuffer.length) {
                        medianScratchBuffer[hBgCount++] = spectrum[b];
                    }
                }
            }
            float hBg = hBgCount > 0 ? computeMedian(medianScratchBuffer, hBgCount) : 1e-6f;
            // Wymóg: wyrazistość względem tła (> 1.6x) oraz bezwzględna energia względem średniej widma (> 5%)
            if (maxHVal / Math.max(1e-6f, hBg) > 1.6f && maxHVal > meanEnergy * 0.05f) {
                harmonicHits++;
                harmonicEnergy += (double) maxHVal * maxHVal;
                if (Math.abs(m - 2.0f) < 0.01f) {
                    has2ndHarmonicHit = true;
                }
            }
        }

        float score = (f0Prominence * 0.4f) + ((float) harmonicHits / multipliers.length * 6.0f);
        float energyRatio = (float) (harmonicEnergy / totalEnergy);

        // Wymóg rotacji HDD:
        // 1. Multi-harmonic comb: harmonicHits >= 2 i score >= 3.0f oraz f0Prominence >= 2.2f
        //    (lub f0Prominence >= 1.8f przy min. 3 harmonicznych i score >= 3.8f)
        // 2. Silny 2-tonowy rotor z f0 i 2f0 (np. autentyczne silniki FDB): harmonicHits >= 1,
        //    f0Prominence >= 3.0f, score >= 3.0f, POD WARUNKIEM że trafienie przypada na 2. harmoniczną (m=2.0)
        boolean multiHitComb = harmonicHits >= 2
                && score >= 3.0f
                && ((f0Prominence >= 2.2f)
                || (f0Prominence >= 1.8f && harmonicHits >= 3 && score >= 3.8f));

        boolean single2ndHarmonicComb = harmonicHits >= 1
                && has2ndHarmonicHit
                && f0Prominence >= 3.0f
                && score >= 3.0f
                && f0Sharpness >= 6.0f;

        boolean detected = (multiHitComb || single2ndHarmonicComb)
                && (energyRatio >= 0.005f || harmonicHits >= 4);

        return new CombScore(detected, score, exactF0Hz, harmonicHits, f0Prominence);
    }

    /**
     * Oblicza medianę z pierwszych 'count' elementów bufora.
     */
    private float computeMedian(float[] buf, int count) {
        if (count <= 0) return 1e-6f;
        if (count == 1) return buf[0];
        Arrays.sort(buf, 0, count);
        if ((count & 1) == 1) {
            return buf[count / 2];
        } else {
            return 0.5f * (buf[count / 2 - 1] + buf[count / 2]);
        }
    }

    /**
     * Oblicza uśrednione widmo amplitudowe z oknem Hann (zgodne z scipy.signal.welch, scaling='spectrum').
     * Wykorzystuje prealokowane tablice w pamięci, eliminując alokację obiektów GC.
     */
    private float[] computeWelchSpectrum(float[] samples, int nFft) {
        int halfFft = nFft / 2 + 1;
        Arrays.fill(avgPowerBuffer, 0f);
        Arrays.fill(avgSpectrumBuffer, 0f);
        int hop = nFft / 2;
        int numSegments = 0;

        for (int offset = 0; offset + nFft <= samples.length; offset += hop) {
            for (int i = 0; i < nFft; i++) {
                realBuffer[i] = samples[offset + i] * hannWindow[i];
                imagBuffer[i] = 0f;
            }

            radix2Fft(realBuffer, imagBuffer, nFft);

            for (int k = 0; k < halfFft; k++) {
                float pwr = realBuffer[k] * realBuffer[k] + imagBuffer[k] * imagBuffer[k];
                avgPowerBuffer[k] += pwr;
            }
            numSegments++;
        }

        if (numSegments > 0) {
            float invSeg = 1.0f / numSegments;
            for (int k = 0; k < halfFft; k++) {
                avgSpectrumBuffer[k] = (float) Math.sqrt(avgPowerBuffer[k] * invSeg);
            }
        }
        return avgSpectrumBuffer;
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

    /**
     * Weryfikacja stacjonarności częstotliwości podstawowej f0 w czasie.
     * Autentyczny silnik HDD wykazuje sztywną mechaniczną stabilność obrotów (wow & flutter < 0.05%, dryf < 0.2 Hz).
     * Głos ludzki (wibrato) oraz syreny charakteryzują się dynamiczną modulacją częstotliwości (FM delta >= 3.0 Hz).
     * Zero-allocation: wykorzystuje prealokowane tablice realBuffer i imagBuffer.
     */
    private boolean verifyTemporalPitchStationarity(float[] samples, float f0Found) {
        if (samples.length < DspConstants.SAMPLE_RATE * 2.5f) {
            return true;
        }

        int nFft = FFT_SIZE; // 4096
        int kMin = Math.max(1, (int) Math.ceil((f0Found - 5.0f) / FREQ_BIN_WIDTH));
        int kMax = Math.min(nFft / 2 - 1, (int) Math.floor((f0Found + 5.0f) / FREQ_BIN_WIDTH));
        if (kMin > kMax) {
            kMin = Math.max(1, Math.round((f0Found - 5.0f) / FREQ_BIN_WIDTH));
            kMax = Math.min(nFft / 2 - 1, Math.round((f0Found + 5.0f) / FREQ_BIN_WIDTH));
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
                realBuffer[i] = samples[offset + i] * hannWindow[i];
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
                float delta = denom != 0.0f ? (gamma - alpha) / denom : 0.0f;
                float fSub = (bestK + delta) * FREQ_BIN_WIDTH;

                if (fSub < minFSub) minFSub = fSub;
                if (fSub > maxFSub) maxFSub = fSub;
                validCount++;
            }
        }

        if (validCount < 3) {
            return true;
        }

        float deltaF = maxFSub - minFSub;
        return deltaF <= 1.5f;
    }
}
