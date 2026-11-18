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
 *    -> Weto dla 'scratching' (indeks 3) i 'clicking' (indeks 1) z wyjątkiem Smart Clipping Bypass
 * 3. Brak wąskopasmowego piku powyżej 1050 Hz lub potwierdzona rotacja wrzeciona
 *    -> Weto dla 'beeping' (indeks 0)
 * 4. Smart Clipping Bypass:
 *    Jeśli sygnał wykazuje ekstremalną dynamikę impulsową (CF >= 4.5) oraz szerokopasmową energię tarcia >4 kHz (highFreqFraction > 0.18),
 *    system traktuje przesterowanie jako dowód katastrofalnej siły uderzenia głowicy (Crash Stop) i dopuszcza klasyfikację awarii.
 * 5. Ochrona pisku stojana (beeping):
 *    Silny, wąskopasmowy ton zablokowanego stojana (1050-3500 Hz) przy braku rotacji wrzeciona
 *    nie jest blokowany przesterowaniem toru mikrofonowego.
 */
public class DspPhysicalGuard {

    private static final int SR = DspConstants.SAMPLE_RATE; // 22050 Hz
    private static final float EPSILON = 1e-12f;

    private static final int FFT_BAND_SIZE = 2048;
    private final float[] hannWindow2048;
    private final float[] fftReal2048;
    private final float[] fftImag2048;

    public DspPhysicalGuard() {
        this.hannWindow2048 = new float[FFT_BAND_SIZE];
        for (int i = 0; i < FFT_BAND_SIZE; i++) {
            hannWindow2048[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / FFT_BAND_SIZE));
        }
        this.fftReal2048 = new float[FFT_BAND_SIZE];
        this.fftImag2048 = new float[FFT_BAND_SIZE];
    }

    /**
     * Struktura przechowująca energie w kluczowych pasmach widmowych obliczona w pojedynczym przebiegu FFT.
     */
    public static class SpectralBands {
        public final float totalEnergy;     // 50 - 10000 Hz
        public final float lowFreqEnergy;   // 50 - 300 Hz
        public final float lowMidEnergy;    // 100 - 1500 Hz
        public final float highFreqEnergy;  // 4000 - 10000 Hz

        public SpectralBands(float totalEnergy, float lowFreqEnergy, float lowMidEnergy, float highFreqEnergy) {
            this.totalEnergy = totalEnergy;
            this.lowFreqEnergy = lowFreqEnergy;
            this.lowMidEnergy = lowMidEnergy;
            this.highFreqEnergy = highFreqEnergy;
        }

        public float getHighFreqFraction() {
            return (totalEnergy > EPSILON) ? (highFreqEnergy / totalEnergy) : 0f;
        }

        public float getLowFreqRatio() {
            return (totalEnergy > EPSILON) ? (lowFreqEnergy / totalEnergy) : 0f;
        }

        public float getLowMidRatio() {
            return (totalEnergy > EPSILON) ? (lowMidEnergy / totalEnergy) : 0f;
        }
    }

    public static class SanitizationResult {
        public final float[] sanitizedProbs;
        public final float crestFactor;
        public final int transientPeakCount;
        public final boolean isAerodynamicFanOrPurifier;
        public final boolean isMicrophoneClipped;
        public final boolean smartClippingBypass;
        public final boolean vetoClicking;
        public final boolean vetoScratching;
        public final boolean vetoBeeping;
        public final String auditLog;

        public SanitizationResult(float[] sanitizedProbs, float crestFactor, int transientPeakCount,
                                  boolean isAerodynamicFanOrPurifier, boolean isMicrophoneClipped,
                                  boolean smartClippingBypass,
                                  boolean vetoClicking, boolean vetoScratching, boolean vetoBeeping,
                                  String auditLog) {
            this.sanitizedProbs = sanitizedProbs;
            this.crestFactor = crestFactor;
            this.transientPeakCount = transientPeakCount;
            this.isAerodynamicFanOrPurifier = isAerodynamicFanOrPurifier;
            this.isMicrophoneClipped = isMicrophoneClipped;
            this.smartClippingBypass = smartClippingBypass;
            this.vetoClicking = vetoClicking;
            this.vetoScratching = vetoScratching;
            this.vetoBeeping = vetoBeeping;
            this.auditLog = auditLog;
        }

        // Konstruktor wstecznie kompatybilny
        public SanitizationResult(float[] sanitizedProbs, float crestFactor, int transientPeakCount,
                                  boolean isAerodynamicFanOrPurifier, boolean isMicrophoneClipped,
                                  boolean vetoClicking, boolean vetoScratching, boolean vetoBeeping,
                                  String auditLog) {
            this(sanitizedProbs, crestFactor, transientPeakCount, isAerodynamicFanOrPurifier,
                 isMicrophoneClipped, false, vetoClicking, vetoScratching, vetoBeeping, auditLog);
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
            return new SanitizationResult(rawProbs, 0f, 0, false, false, false, false, false, false, "Brak danych");
        }

        // 1. Usunięcie DC offsetu z sygnału fali
        float[] samples = removeDcOffset(rawSamples);

        // 2. Obliczenie cech fizycznych
        float rms = computeRms(samples);

        // Odrzucenie ciszy / braku sygnału (ADR-001 Invariant: RMS < 0.002f)
        if (rms < DspConstants.RMS_SILENCE_THRESHOLD) {
            float[] silentProbs = new float[]{0.0f, 0.0f, 1.0f, 0.0f};
            return new SanitizationResult(
                    silentProbs,
                    0f,
                    0,
                    false,
                    false,
                    false,
                    rawProbs[1] > 0.05f,
                    rawProbs[3] > 0.05f,
                    rawProbs[0] > 0.05f,
                    String.format(java.util.Locale.US,
                            "[Cisza / brak sygnału: RMS=%.5f < %.4f - wymuszono zero prawdopodobieństwa awarii]",
                            rms, DspConstants.RMS_SILENCE_THRESHOLD)
            );
        }
        float peak = computePeak(samples);
        float crestFactor = peak / Math.max(rms, EPSILON);

        int transientCount = countTransientPeaks(samples, SR);

        // Zoptymalizowana pojedyncza transformata FFT Radix-2 dla wszystkich pasm widmowych
        SpectralBands bands = computeSpectralBands(samples);
        float highFreqFraction = bands.getHighFreqFraction();
        float lowFreqRatio = bands.getLowFreqRatio();

        boolean isFanOrPurifier = isAerodynamicFanOrPurifier(samples, crestFactor, transientCount, bands);
        boolean isClipped = isMicrophoneClipped(samples, peak, crestFactor);

        // Szum miejski/otoczenia o silnej dominacji basu i braku wysokich częstotliwości
        boolean isCityTrafficNoise = (lowFreqRatio > 0.75f && highFreqFraction < 0.035f);

        // Smart Clipping Bypass:
        // Jeśli sygnał wykazuje ekstremalną dynamikę impulsową (CF >= 4.5) oraz szerokopasmową energię tarcia > 4 kHz (highFreqFraction > 0.18),
        // przesterowanie przetwornika ADC jest dowodem potężnego uderzenia mechanicznego głowicy (Crash Stop / Head Slap).
        // W takim przypadku weto clippingu zostaje ominięte dla awarii mechanicznych (clicking i scratching).
        boolean smartClippingBypass = (crestFactor >= 4.5f && highFreqFraction > 0.18f);
        boolean effectiveClippingVeto = isClipped && !smartClippingBypass;

        float[] probs = Arrays.copyOf(rawProbs, rawProbs.length);
        boolean vetoClicking = false;
        boolean vetoScratching = false;
        boolean vetoBeeping = false;
        StringBuilder audit = new StringBuilder();

        // Indeksy klas: 0: beeping, 1: clicking, 2: normal, 3: scratching

        // WETO 1: CLICKING (indeks 1)
        // Autentyczny Click of Death wymaga wyrazistych uderzeń (transientCount >= 2) oraz wysokiego Crest Factor (>= 3.5).
        // Jednostajny szum wentylatora ani basowy szum miasta NIE mogą generować clicking!
        // Przesterowanie mikrofonu wetuje klikanie, chyba że aktywny jest Smart Clipping Bypass (katastrofalny Crash Stop).
        if (crestFactor < 3.5f || transientCount < 2 || isFanOrPurifier || effectiveClippingVeto || isCityTrafficNoise) {
            if (probs[1] > 0.05f) {
                vetoClicking = true;
                float confiscated = probs[1] - 0.01f;
                probs[1] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto CLICKING: CF=%.2f, transients=%d, fan=%b, clipped=%b (bypass=%b), cityNoise=%b] ",
                        crestFactor, transientCount, isFanOrPurifier, isClipped, smartClippingBypass, isCityTrafficNoise));
            }
        } else if (isClipped && smartClippingBypass) {
            audit.append(String.format(java.util.Locale.US,
                    "[Smart Clipping Bypass dla CLICKING: CF=%.2f, highFreq=%.1f%% - potwierdzono uderzenie Crash Stop] ",
                    crestFactor, highFreqFraction * 100.0f));
        }

        // WETO 2: SCRATCHING (indeks 3)
        // Fałszywe tarcie głowic powstaje przy nieuszkodzonym przesterowaniu mikrofonu, szumie miasta lub szumie powietrza.
        // Jeśli sygnał posiada cechy katastrofalnego tarcia głowic o talerz (CF >= 4.5 i highFreq > 18%), omijamy weto clippingu.
        if (effectiveClippingVeto || isCityTrafficNoise || (isFanOrPurifier && highFreqFraction < 0.15f)) {
            if (probs[3] > 0.05f) {
                vetoScratching = true;
                float confiscated = probs[3] - 0.01f;
                probs[3] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto SCRATCHING: clipped=%b (bypass=%b), cityNoise=%b, highFreq=%.1f%%] ",
                        isClipped, smartClippingBypass, isCityTrafficNoise, highFreqFraction * 100.0f));
            }
        } else if (isClipped && smartClippingBypass) {
            audit.append(String.format(java.util.Locale.US,
                    "[Smart Clipping Bypass dla SCRATCHING: CF=%.2f, highFreq=%.1f%% - potwierdzono ostre tarcie głowic] ",
                    crestFactor, highFreqFraction * 100.0f));
        }

        // WETO 3: BEEPING (indeks 0)
        // Dysk piszczy zablokowanym silnikiem wyłącznie, gdy silnik NIE wiruje i występuje ostry pisk > 1050 Hz.
        // Ochrona pisku stojana: nie blokujemy klasy beeping samym warunkiem isClipped,
        // jeżeli RpmHarmonicsAnalyzer potwierdził brak rotacji oraz silny ton stojana (hasStallTone == true w paśmie 1050-3500 Hz).
        boolean hasNormalRotation = (rpmRes != null && rpmRes.hasRotation);
        boolean hasStallTone = (rpmRes != null && rpmRes.hasStallBeep && rpmRes.stallBeepFreqHz >= 1050.0f);
        boolean isStrongStallTone = hasStallTone && !hasNormalRotation;

        boolean vetoBeepingCondition;
        if (isStrongStallTone) {
            // Silny, wąskopasmowy pisk zablokowanego stojana bez rotacji:
            // IGNORUJEMY clipping toru mikrofonowego!
            vetoBeepingCondition = (isCityTrafficNoise && !hasStallTone);
        } else {
            vetoBeepingCondition = hasNormalRotation || !hasStallTone || isClipped || (isCityTrafficNoise && !hasStallTone);
        }

        if (vetoBeepingCondition) {
            if (probs[0] > 0.05f) {
                vetoBeeping = true;
                float confiscated = probs[0] - 0.01f;
                probs[0] = 0.01f;
                probs[2] += confiscated;
                audit.append(String.format(java.util.Locale.US,
                        "[Weto BEEPING: rot=%b, stallTone=%b (strong=%b), clipped=%b, cityNoise=%b] ",
                        hasNormalRotation, hasStallTone, isStrongStallTone, isClipped, isCityTrafficNoise));
            }
        }

        // 3. Renormalizacja prawdopodobieństw do 1.0
        float sum = 0f;
        for (float p : probs) {
            sum += Math.max(p, 0f);
        }
        if (sum > EPSILON) {
            float invSum = 1.0f / sum;
            for (int i = 0; i < probs.length; i++) {
                probs[i] = Math.max(probs[i], 0f) * invSum;
            }
        }

        return new SanitizationResult(
                probs,
                crestFactor,
                transientCount,
                isFanOrPurifier,
                isClipped,
                smartClippingBypass,
                vetoClicking,
                vetoScratching,
                vetoBeeping,
                audit.toString().trim()
        );
    }

    /**
     * Oblicza energie w 4 kluczowych pasmach częstotliwości w pojedynczym przebiegu FFT Radix-2:
     * - totalEnergy (50 - 10000 Hz)
     * - lowFreqEnergy (50 - 300 Hz)
     * - lowMidEnergy (100 - 1500 Hz)
     * - highFreqEnergy (4000 - 10000 Hz)
     */
    public synchronized SpectralBands computeSpectralBands(float[] zeroMeanSamples) {
        if (zeroMeanSamples == null || zeroMeanSamples.length < FFT_BAND_SIZE) {
            return new SpectralBands(0f, 0f, 0f, 0f);
        }

        int step = FFT_BAND_SIZE / 2; // 1024
        int nSteps = (zeroMeanSamples.length - FFT_BAND_SIZE) / step + 1;
        if (nSteps <= 0) {
            return new SpectralBands(0f, 0f, 0f, 0f);
        }

        float binWidth = (float) SR / FFT_BAND_SIZE; // 22050 / 2048 ≈ 10.7666 Hz
        int bin50 = Math.max(0, Math.round(50.0f / binWidth));
        int bin300 = Math.min(FFT_BAND_SIZE / 2, Math.round(300.0f / binWidth));
        int bin100 = Math.max(0, Math.round(100.0f / binWidth));
        int bin1500 = Math.min(FFT_BAND_SIZE / 2, Math.round(1500.0f / binWidth));
        int bin4000 = Math.max(0, Math.round(4000.0f / binWidth));
        int bin10000 = Math.min(FFT_BAND_SIZE / 2, Math.round(10000.0f / binWidth));

        double accTotal = 0.0;
        double accLow = 0.0;
        double accLowMid = 0.0;
        double accHigh = 0.0;

        for (int s = 0; s < nSteps; s++) {
            int offset = s * step;
            for (int i = 0; i < FFT_BAND_SIZE; i++) {
                fftReal2048[i] = zeroMeanSamples[offset + i] * hannWindow2048[i];
                fftImag2048[i] = 0f;
            }

            fftRadix2(fftReal2048, fftImag2048, FFT_BAND_SIZE);

            int halfFft = FFT_BAND_SIZE / 2;
            for (int k = 0; k <= halfFft; k++) {
                double pwr = (double) fftReal2048[k] * fftReal2048[k] + (double) fftImag2048[k] * fftImag2048[k];
                if (k >= bin50 && k <= bin10000) {
                    accTotal += pwr;
                }
                if (k >= bin50 && k <= bin300) {
                    accLow += pwr;
                }
                if (k >= bin100 && k <= bin1500) {
                    accLowMid += pwr;
                }
                if (k >= bin4000 && k <= bin10000) {
                    accHigh += pwr;
                }
            }
        }

        double norm = (double) nSteps * FFT_BAND_SIZE;
        return new SpectralBands(
                (float) (accTotal / norm),
                (float) (accLow / norm),
                (float) (accLowMid / norm),
                (float) (accHigh / norm)
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
     * Sprawdza, czy sygnał to ciągły szum aerodynamiczny wentylatora, turbiny, klimatyzacji lub oczyszczacza powietrza
     * z wykorzystaniem prealokowanych energii pasmowych SpectralBands.
     */
    public static boolean isAerodynamicFanOrPurifier(float[] zeroMeanSamples, float crestFactor, int transientCount, SpectralBands bands) {
        if (crestFactor >= 6.2f || transientCount >= 4) {
            return false;
        }
        if (bands == null || bands.totalEnergy < EPSILON) {
            return false;
        }

        float highRatio = bands.getHighFreqFraction();
        float lowMidRatio = bands.getLowMidRatio();

        // Szum turbulentny powietrza / otoczenia: wysokie pasmo >4 kHz jest znikome (< 4%),
        // a energia skupiona jest w dolnym i średnim paśmie.
        if (highRatio < 0.04f && (lowMidRatio > 0.50f || crestFactor < 5.5f)) {
            return true;
        }

        // Standardowy wentylator/oczyszczacz
        return (lowMidRatio > 0.60f && highRatio < 0.12f && crestFactor < 4.5f && transientCount < 2);
    }

    /**
     * Przeciążenie zachowujące kompatybilność wsteczną.
     */
    public static boolean isAerodynamicFanOrPurifier(float[] zeroMeanSamples, float crestFactor, int transientCount) {
        if (crestFactor >= 6.2f || transientCount >= 4) {
            return false;
        }
        float lowMidEnergy = computeBandEnergy(zeroMeanSamples, SR, 100.0f, 1500.0f);
        float highEnergy = computeBandEnergy(zeroMeanSamples, SR, 4000.0f, 10000.0f);
        float totalEnergy = computeBandEnergy(zeroMeanSamples, SR, 50.0f, 10000.0f);

        if (totalEnergy < EPSILON) return false;

        float highRatio = highEnergy / totalEnergy;
        float lowMidRatio = lowMidEnergy / totalEnergy;

        if (highRatio < 0.04f && (lowMidRatio > 0.50f || crestFactor < 5.5f)) {
            return true;
        }

        return (lowMidRatio > 0.60f && highRatio < 0.12f && crestFactor < 4.5f && transientCount < 2);
    }

    /**
     * Sprawdza, czy wystąpiło przesterowanie toru mikrofonu (Hard Clipping):
     * - Wymaga minimalnej amplitudy bezwzględnej nasycenia przetwornika ADC (peak >= 0.35f)
     * - Ponad 2% próbek osiąga >= 0.985 wartości szczytowej
     * - Lub występują spłaszczenia fali (flat run >= 3) przy niskim Crest Factor (< 2.5).
     */
    public static boolean isMicrophoneClipped(float[] zeroMeanSamples, float peak, float crestFactor) {
        if (zeroMeanSamples == null || zeroMeanSamples.length == 0 || peak < 0.35f) {
            return false;
        }

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
    public static float computeHighFrequencyEnergyFraction(float[] samples, int sr, float cutoffHz) {
        float highEnergy = computeBandEnergy(samples, sr, cutoffHz, 10000.0f);
        float totalEnergy = computeBandEnergy(samples, sr, 50.0f, 10000.0f);
        if (totalEnergy < EPSILON) return 0f;
        return highEnergy / totalEnergy;
    }

    /**
     * Oblicza przybliżoną energię w określonym paśmie częstotliwości za pomocą transformaty FFT Radix-2.
     */
    public static float computeBandEnergy(float[] samples, int sr, float fLow, float fHigh) {
        if (samples == null || samples.length < FFT_BAND_SIZE) return 0f;
        int nFft = FFT_BAND_SIZE;
        int step = nFft / 2;
        int nSteps = (samples.length - nFft) / step + 1;
        if (nSteps <= 0) return 0f;

        float binWidth = (float) sr / nFft;
        int binLow = Math.max(0, (int) (fLow / binWidth));
        int binHigh = Math.min(nFft / 2, (int) (fHigh / binWidth));

        float[] window = new float[nFft];
        for (int i = 0; i < nFft; i++) {
            window[i] = 0.5f * (1.0f - (float) Math.cos(2.0 * Math.PI * i / nFft));
        }

        float[] real = new float[nFft];
        float[] imag = new float[nFft];
        double totalBandPower = 0.0;

        for (int s = 0; s < nSteps; s++) {
            int offset = s * step;
            for (int i = 0; i < nFft; i++) {
                real[i] = samples[offset + i] * window[i];
                imag[i] = 0f;
            }

            fftRadix2(real, imag, nFft);

            for (int k = binLow; k <= binHigh; k++) {
                totalBandPower += (double) real[k] * real[k] + (double) imag[k] * imag[k];
            }
        }

        return (float) (totalBandPower / ((double) nSteps * nFft));
    }

    /**
     * Szybka transformata Fouriera Radix-2 Cooley-Tukey (in-place).
     */
    private static void fftRadix2(float[] real, float[] imag, int n) {
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

                for (int k = 0; k < half; k++) {
                    int pos = i + k;
                    int pair = pos + half;

                    float v_r = real[pair] * w_r - imag[pair] * w_i;
                    float v_i = real[pair] * w_i + imag[pair] * w_r;

                    float u_r = real[pos];
                    float u_i = imag[pos];

                    real[pos] = u_r + v_r;
                    imag[pos] = u_i + v_i;
                    real[pair] = u_r - v_r;
                    imag[pair] = u_i - v_i;

                    float next_w_r = w_r * wlen_r - w_i * wlen_i;
                    float next_w_i = w_r * wlen_i + w_i * wlen_r;
                    w_r = next_w_r;
                    w_i = next_w_i;
                }
            }
        }
    }
}
