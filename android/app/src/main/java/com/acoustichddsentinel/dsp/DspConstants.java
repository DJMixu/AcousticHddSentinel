package com.acoustichddsentinel.dsp;

/**
 * Niezmienniki techniczne parametrów cyfrowego przetwarzania sygnałów (DSP).
 * Zgodne z ADR-001 (Single Source of Truth dla AcousticHddSentinel).
 */
public final class DspConstants {

    private DspConstants() {
        // Blokada instancjonowania
    }

    /** Docelowa częstotliwość próbkowania sygnału audio (Hz). */
    public static final int SAMPLE_RATE = 22050;

    /** Alternatywna częstotliwość próbkowania sprzętu Android (Hz) przed decymacją. */
    public static final int FALLBACK_SAMPLE_RATE = 44100;

    /** Rozmiar okna szybkiej transformaty Fouriera (FFT). */
    public static final int N_FFT = 1024;

    /** Krok przesunięcia okna FFT (Hop length) - krok ~11.6 ms kluczowy dla transjentów. */
    public static final int HOP_LENGTH = 256;

    /** Liczba filtrów w pasmowym banku Mel. */
    public static final int N_MELS = 128;

    /** Dolna granica częstotliwości pasma Mel (Hz). */
    public static final float FMIN = 50.0f;

    /** Górna granica częstotliwości pasma Mel (Hz) - pasmo tarcia głowic i pisków. */
    public static final float FMAX = 10000.0f;

    /** Długość pojedynczego wycinka analitycznego w sekundach. */
    public static final float CHUNK_DURATION_SEC = 3.0f;

    /** Liczba próbek audio w wycinku 3.0s (3.0 * 22050 = 66150). */
    public static final int CHUNK_SAMPLES = (int) (CHUNK_DURATION_SEC * SAMPLE_RATE);

    /** Docelowa liczba ramek czasowych spektrogramu dla wycinka 3.0s. */
    public static final int TIME_STEPS = 259;

    /** Liczba kanałów tensora wejściowego modelu 2D CNN (Log-Mel, Delta, Delta-Delta). */
    public static final int NUM_CHANNELS = 3;

    /** Rozmiar spłaszczonego tensora pojedynczego wycinka (1 * 3 * 128 * 259 = 99456). */
    public static final int TENSOR_FLAT_SIZE = NUM_CHANNELS * N_MELS * TIME_STEPS;

    /** Szerokość okna różniczkowego dla wyznaczania kanałów Delta i Delta-Delta. */
    public static final int DELTA_WIDTH = 9;

    /** Próg bramki ciszy (RMS Gate) - wartości poniżej oznaczają brak kontaktu z dyskiem. */
    public static final float RMS_SILENCE_THRESHOLD = 0.002f;

    /** Zakres dynamiki w skali dB (top_db w librosa.power_to_db). */
    public static final float TOP_DB = 80.0f;

    /** Stała numeryczna zabezpieczająca przed dzieleniem przez zero. */
    public static final float EPSILON = 1e-6f;
}
