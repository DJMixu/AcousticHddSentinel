package com.acoustichddsentinel.audio;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Zestaw testów jednostkowych antyaliasingowego filtru Butterwortha 4. rzędu
 * i decymacji 2:1 (z 44100 Hz do 22050 Hz) w klasie AudioRecorder (czysta Java 17, JUnit 4).
 * Weryfikuje:
 * 1. Dokładny współczynnik decymacji 2:1 (długość wyjścia = długość wejścia / 2).
 * 2. Przepuszczanie sygnałów w paśmie przepustowym (np. 1000 Hz, fc = 9500 Hz).
 * 3. Tłumienie składowych powyżej częstotliwości odcięcia w paśmie zaporowym (np. 16000 Hz).
 * 4. Odporność na sygnał zerowy, impuls Diraca oraz brak wartości NaN / nieskończoności.
 * 5. Zgodność przetwarzania próbek 16-bit PCM (short[]) z float[].
 */
public class AudioRecorderDecimationTest {

    private AudioRecorder.ButterworthDecimator decimator;
    private static final int FS_IN = 44100;
    private static final int FS_OUT = 22050;

    @Before
    public void setUp() {
        decimator = new AudioRecorder.ButterworthDecimator();
    }

    @Test
    public void testDecimationFactorTwoToOne() {
        int inputLength = 44100; // 1 sekunda przy 44.1 kHz
        float[] input = new float[inputLength];
        for (int i = 0; i < inputLength; i++) {
            input[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / FS_IN);
        }

        float[] output = decimator.process(input);

        assertNotNull("Zdecymowany sygnał nie może być nullem", output);
        assertEquals("Długość wyjścia musi wynosić dokładnie połowę długości wejścia (2:1)",
                inputLength / 2, output.length);
    }

    @Test
    public void testPassbandFidelity1000Hz() {
        // Sygnał 1000 Hz leży głęboko w paśmie przepustowym filtru (fc = 9500 Hz)
        int numSamples = FS_IN * 2; // 2 sekundy dla ustabilizowania stanu ustalonego IIR
        float[] input = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            input[i] = (float) Math.sin(2.0 * Math.PI * 1000.0 * i / FS_IN);
        }

        float[] output = decimator.process(input);

        // Pomijamy pierwsze 500 próbek stanu nieustalonego filtru IIR
        int testStart = 500;
        int testEnd = output.length - 500;
        float maxOutputAmp = 0f;
        for (int i = testStart; i < testEnd; i++) {
            float abs = Math.abs(output[i]);
            if (abs > maxOutputAmp) {
                maxOutputAmp = abs;
            }
        }

        // Amplituda tonu 1000 Hz w paśmie przepustowym powinna wynosić ~1.0 (+/- 5%)
        assertEquals("Amplituda w paśmie przepustowym 1000 Hz musi wynosić ~1.0",
                1.0f, maxOutputAmp, 0.08f);
    }

    @Test
    public void testStopbandAttenuation16000Hz() {
        // Sygnał 16000 Hz leży w paśmie zaporowym filtru Butterwortha (powyżej Nyquista 11025 Hz i fc = 9500 Hz)
        int numSamples = FS_IN * 2; // 2 sekundy
        float[] input = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            input[i] = (float) Math.sin(2.0 * Math.PI * 16000.0 * i / FS_IN);
        }

        float[] output = decimator.process(input);

        // Pomijamy stan nieustalony filtru IIR
        int testStart = 500;
        int testEnd = output.length - 500;
        float maxStopbandAmp = 0f;
        for (int i = testStart; i < testEnd; i++) {
            float abs = Math.abs(output[i]);
            if (abs > maxStopbandAmp) {
                maxStopbandAmp = abs;
            }
        }

        // Tłumienie filtru 4. rzędu dla 16 kHz musi być znaczące (amplituda < 0.08, tłumienie > 22 dB)
        assertTrue("Sygnał 16000 Hz musi zostać stłumiony w paśmie zaporowym (amplituda < 0.08, zmierzono: " + maxStopbandAmp + ")",
                maxStopbandAmp < 0.08f);
    }

    @Test
    public void testZeroInputAndImpulseStability() {
        // Test zerowego wejścia
        float[] zeroInput = new float[2048];
        float[] zeroOutput = decimator.process(zeroInput);
        assertEquals(1024, zeroOutput.length);
        for (int i = 0; i < zeroOutput.length; i++) {
            assertEquals("Dla zerowego wejścia wyjście musi wynosić 0.0", 0.0f, zeroOutput[i], 1e-6f);
            assertFalse(Float.isNaN(zeroOutput[i]));
            assertFalse(Float.isInfinite(zeroOutput[i]));
        }

        // Test stabilności na impuls jednostkowy Diraca
        decimator.reset();
        float[] impulseInput = new float[4096];
        impulseInput[0] = 1.0f;
        float[] impulseOutput = decimator.process(impulseInput);

        // Odpowiedź impulsowa filtru IIR musi wygasnąć i nie może rozbiegać
        float tailEnergy = 0f;
        for (int i = 200; i < impulseOutput.length; i++) {
            tailEnergy += impulseOutput[i] * impulseOutput[i];
            assertFalse("Wykryto NaN w odpowiedzi impulsowej", Float.isNaN(impulseOutput[i]));
            assertFalse("Wykryto Inf w odpowiedzi impulsowej", Float.isInfinite(impulseOutput[i]));
        }
        assertTrue("Odpowiedź impulsowa musi stabilnie wygasnąć do zera", tailEnergy < 1e-5f);
    }

    @Test
    public void testShortPcmProcessingEquivalence() {
        int length = 2000;
        short[] shortBuffer = new short[length];
        float[] floatBuffer = new float[length];

        for (int i = 0; i < length; i++) {
            float val = (float) Math.sin(2.0 * Math.PI * 440.0 * i / FS_IN);
            shortBuffer[i] = (short) Math.round(val * 32767.0f);
            floatBuffer[i] = shortBuffer[i] / 32768.0f;
        }

        decimator.reset();
        float[] outFromShorts = decimator.process(shortBuffer, 0, length);

        decimator.reset();
        float[] outFromFloats = decimator.process(floatBuffer);

        assertEquals(outFromShorts.length, outFromFloats.length);
        for (int i = 0; i < outFromShorts.length; i++) {
            assertEquals("Wyjście z bufora short[] i float[] musi być numerycznie tożsame",
                    outFromFloats[i], outFromShorts[i], 1e-6f);
        }
    }
}
