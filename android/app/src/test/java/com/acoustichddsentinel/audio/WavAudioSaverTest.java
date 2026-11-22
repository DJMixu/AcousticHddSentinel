package com.acoustichddsentinel.audio;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe narzędzia archiwizacji nagrań audio WavAudioSaver.
 * Weryfikuje strukturę nagłówka RIFF WAV 16-bit PCM (22050 Hz, Mono),
 * precyzję kwantyzacji próbek, mechanizmy clamping oraz usuwanie plików.
 */
public class WavAudioSaverTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testWavConstants() {
        assertEquals("Domyślna częstotliwość próbkowania WAV musi wynosić 22050 Hz (ADR-001)",
                22050, WavAudioSaver.DEFAULT_SAMPLE_RATE);
        assertEquals("Podkatalog nagrań musi mieć nazwę recordings",
                "recordings", WavAudioSaver.RECORDINGS_SUBDIR);
    }

    @Test
    public void testWavFileHeaderAndDataIntegrity() throws IOException {
        File wavFile = tempFolder.newFile("test_sample.wav");
        int sampleRate = 22050;
        int numSamples = 22050; // 1 sekunda audio
        float[] samples = new float[numSamples];

        // Wygenerowanie sygnału testowego: sinusoida 440 Hz (A4)
        for (int i = 0; i < numSamples; i++) {
            samples[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / sampleRate) * 0.8f;
        }

        WavAudioSaver.writeWavFile(wavFile, samples, sampleRate);

        assertTrue("Plik WAV musi istnieć na dysku", wavFile.exists());
        int expectedFileSize = 44 + numSamples * 2;
        assertEquals("Rozmiar pliku WAV musi wynosić 44B nagłówka + próbki 16-bit",
                expectedFileSize, wavFile.length());

        byte[] fileBytes = new byte[expectedFileSize];
        try (FileInputStream fis = new FileInputStream(wavFile)) {
            int read = fis.read(fileBytes);
            assertEquals(expectedFileSize, read);
        }

        ByteBuffer buffer = ByteBuffer.wrap(fileBytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        // 1. Sprawdzenie nagłówka RIFF
        byte[] riffTag = new byte[4];
        buffer.get(riffTag);
        assertEquals("RIFF", new String(riffTag, StandardCharsets.US_ASCII));

        int chunkSize = buffer.getInt();
        assertEquals(expectedFileSize - 8, chunkSize);

        byte[] waveTag = new byte[4];
        buffer.get(waveTag);
        assertEquals("WAVE", new String(waveTag, StandardCharsets.US_ASCII));

        // 2. Sprawdzenie sub-chunku "fmt "
        byte[] fmtTag = new byte[4];
        buffer.get(fmtTag);
        assertEquals("fmt ", new String(fmtTag, StandardCharsets.US_ASCII));

        int subchunk1Size = buffer.getInt();
        assertEquals(16, subchunk1Size); // 16 dla PCM

        short audioFormat = buffer.getShort();
        assertEquals(1, audioFormat); // 1 = Linear PCM

        short numChannels = buffer.getShort();
        assertEquals(1, numChannels); // Mono

        int actualSampleRate = buffer.getInt();
        assertEquals(22050, actualSampleRate);

        int byteRate = buffer.getInt();
        assertEquals(44100, byteRate); // 22050 * 1 * 2

        short blockAlign = buffer.getShort();
        assertEquals(2, blockAlign); // 1 * 2

        short bitsPerSample = buffer.getShort();
        assertEquals(16, bitsPerSample);

        // 3. Sprawdzenie sub-chunku "data"
        byte[] dataTag = new byte[4];
        buffer.get(dataTag);
        assertEquals("data", new String(dataTag, StandardCharsets.US_ASCII));

        int dataSize = buffer.getInt();
        assertEquals(numSamples * 2, dataSize);

        // 4. Weryfikacja odczytanych próbek PCM
        for (int i = 0; i < numSamples; i++) {
            short pcmSample = buffer.getShort();
            float decodedSample = (float) pcmSample / 32767.0f;
            assertEquals("Próbka zdekodowana z WAV musi odpowiadać próbce wejściowej w granicach kwantyzacji 16-bit",
                    samples[i], decodedSample, 1.0f / 32767.0f + 1e-4f);
        }
    }

    @Test
    public void testClampingOfExtremeValues() throws IOException {
        File wavFile = tempFolder.newFile("test_clamping.wav");
        float[] samples = new float[]{1.5f, -1.8f, 0.0f, 1.0f, -1.0f};

        WavAudioSaver.writeWavFile(wavFile, samples, 22050);

        byte[] bytes = new byte[44 + 5 * 2];
        try (FileInputStream fis = new FileInputStream(wavFile)) {
            int r = fis.read(bytes);
            assertEquals(bytes.length, r);
        }

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.position(44); // Przejście do danych

        short s0 = buffer.getShort();
        short s1 = buffer.getShort();
        short s2 = buffer.getShort();
        short s3 = buffer.getShort();
        short s4 = buffer.getShort();

        assertEquals("Wartość > 1.0 powinna być przycięta do 32767", 32767, s0);
        assertEquals("Wartość < -1.0 powinna być przycięta do -32767", -32767, s1);
        assertEquals("Wartość 0.0 powinna wynosić 0", 0, s2);
        assertEquals("Wartość 1.0 powinna wynosić 32767", 32767, s3);
        assertEquals("Wartość -1.0 powinna wynosić -32767", -32767, s4);
    }

    @Test
    public void testDeleteAudioFile() throws IOException {
        File wavFile = tempFolder.newFile("to_delete.wav");
        assertTrue(wavFile.exists());

        boolean deleted = WavAudioSaver.deleteAudioFile(wavFile.getAbsolutePath());
        assertTrue("deleteAudioFile powinno zwrócić true dla istniejącego pliku", deleted);
        assertFalse("Plik nie powinien już istnieć na dysku", wavFile.exists());

        boolean deleteAgain = WavAudioSaver.deleteAudioFile(wavFile.getAbsolutePath());
        assertFalse("deleteAudioFile powinno zwrócić false dla nieistniejącego pliku", deleteAgain);

        assertFalse(WavAudioSaver.deleteAudioFile(null));
        assertFalse(WavAudioSaver.deleteAudioFile("   "));
    }

    @Test
    public void testSaveNullOrEmptyInputsReturnsNull() {
        assertNull("saveAudioChunkToWav z null chunk musi zwrócić null",
                WavAudioSaver.saveAudioChunkToWav(null, null, "normal"));

        AudioChunk emptyChunk = new AudioChunk(0, null, 22050, 0.0f, System.currentTimeMillis());
        assertNull("saveAudioChunkToWav z pustymi próbkami musi zwrócić null",
                WavAudioSaver.saveAudioChunkToWav(null, emptyChunk, "normal"));

        assertNull("saveSamplesToWav z pustą tablicą próbek musi zwrócić null",
                WavAudioSaver.saveSamplesToWav(null, new float[0], 22050, "normal"));
    }
}
