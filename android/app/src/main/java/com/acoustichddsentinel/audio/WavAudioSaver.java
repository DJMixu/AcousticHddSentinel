package com.acoustichddsentinel.audio;

import android.content.Context;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Narzędzie archiwizacji nagrań audio do plików RIFF WAV (16-bit PCM, 22050 Hz, Mono).
 * Umożliwia trwały zapis reprezentatywnych wycinków audio dla historii badań HDD,
 * odsłuch próbek oraz bezpieczne usuwanie z pamięci masowej.
 */
public final class WavAudioSaver {

    private static final String TAG = "WavAudioSaver";
    public static final String RECORDINGS_SUBDIR = "recordings";
    public static final int DEFAULT_SAMPLE_RATE = 22050;

    private WavAudioSaver() {
    }

    /**
     * Zapisuje wycinek AudioChunk do pliku WAV w katalogu dedykowanym aplikacji.
     *
     * @param context        kontekst aplikacji Android
     * @param chunk          wycinek audio do zapisania
     * @param classification etykieta klasy (np. "normal", "clicking", "beeping", "scratching")
     * @return bezwzględna ścieżka do zapisanego pliku WAV lub null w przypadku błędu
     */
    public static String saveAudioChunkToWav(Context context, AudioChunk chunk, String classification) {
        if (chunk == null || chunk.getSamples() == null) {
            Log.e(TAG, "Nie można zapisać WAV - AudioChunk lub próbki są puste.");
            return null;
        }
        return saveSamplesToWav(context, chunk.getSamples(), chunk.getSampleRate(), classification);
    }

    /**
     * Zapisuje tablicę próbek float[] do pliku WAV w katalogu dedykowanym aplikacji.
     *
     * @param context        kontekst aplikacji Android
     * @param samples        próbki audio w formacie znormalizowanym [-1.0, 1.0]
     * @param sampleRate     częstotliwość próbkowania (np. 22050 Hz)
     * @param classification etykieta klasy
     * @return bezwzględna ścieżka do zapisanego pliku WAV lub null w przypadku błędu
     */
    public static String saveSamplesToWav(Context context, float[] samples, int sampleRate, String classification) {
        if (context == null || samples == null || samples.length == 0) {
            Log.e(TAG, "Błędne parametry wejściowe do zapisu pliku WAV.");
            return null;
        }

        File recordingsDir = getRecordingsDirectory(context);
        if (!recordingsDir.exists() && !recordingsDir.mkdirs()) {
            Log.e(TAG, "Nie udało się utworzyć katalogu nagrań: " + recordingsDir.getAbsolutePath());
        }

        String safeClass = (classification != null && !classification.trim().isEmpty())
                ? classification.trim().toLowerCase(Locale.US).replaceAll("[^a-z0-9_]", "")
                : "unknown";

        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US);
        String timestampStr = sdf.format(new Date());
        String fileName = "rec_" + timestampStr + "_" + safeClass + ".wav";
        File targetFile = new File(recordingsDir, fileName);

        // Zabezpieczenie przed kolizją nazw przy szybkich wielokrotnych diagnozach
        int counter = 1;
        while (targetFile.exists()) {
            fileName = "rec_" + timestampStr + "_" + counter + "_" + safeClass + ".wav";
            targetFile = new File(recordingsDir, fileName);
            counter++;
        }

        try {
            writeWavFile(targetFile, samples, sampleRate);
            Log.i(TAG, "Zapisano nagranie WAV: " + targetFile.getAbsolutePath() + " (" + targetFile.length() + " bajtów)");
            return targetFile.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "Błąd podczas zapisu pliku WAV: " + targetFile.getAbsolutePath(), e);
            return null;
        }
    }

    /**
     * Zwraca katalog do przechowywania nagrań dźwiękowych (priorytet: ExternalFilesDir, fallback: FilesDir).
     */
    public static File getRecordingsDirectory(Context context) {
        File baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) {
            baseDir = context.getFilesDir();
        }
        return new File(baseDir, RECORDINGS_SUBDIR);
    }

    /**
     * Niskopoziomowy zapis tablicy float[] jako 16-bit PCM RIFF WAV.
     * Skaluje próbki z zakresu [-1.0, 1.0] do signed 16-bit short [-32768, 32767] z soft-clampingiem.
     *
     * @param targetFile docelowy plik WAV
     * @param samples    próbki audio
     * @param sampleRate częstotliwość próbkowania
     * @throws IOException w przypadku błędu wejścia/wyjścia
     */
    public static void writeWavFile(File targetFile, float[] samples, int sampleRate) throws IOException {
        int numSamples = samples.length;
        int numChannels = 1; // Mono
        int bitsPerSample = 16;
        int byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        short blockAlign = (short) (numChannels * (bitsPerSample / 8));
        int dataSize = numSamples * numChannels * (bitsPerSample / 8);
        int chunkSize = 36 + dataSize;

        ByteBuffer header = ByteBuffer.allocate(44);
        header.order(ByteOrder.LITTLE_ENDIAN);

        // RIFF chunk descriptor
        header.put((byte) 'R');
        header.put((byte) 'I');
        header.put((byte) 'F');
        header.put((byte) 'F');
        header.putInt(chunkSize);
        header.put((byte) 'W');
        header.put((byte) 'A');
        header.put((byte) 'V');
        header.put((byte) 'E');

        // "fmt " sub-chunk
        header.put((byte) 'f');
        header.put((byte) 'm');
        header.put((byte) 't');
        header.put((byte) ' ');
        header.putInt(16); // Subchunk1Size dla PCM = 16
        header.putShort((short) 1); // AudioFormat: 1 = PCM (linear quantization)
        header.putShort((short) numChannels);
        header.putInt(sampleRate);
        header.putInt(byteRate);
        header.putShort(blockAlign);
        header.putShort((short) bitsPerSample);

        // "data" sub-chunk
        header.put((byte) 'd');
        header.put((byte) 'a');
        header.put((byte) 't');
        header.put((byte) 'a');
        header.putInt(dataSize);

        try (FileOutputStream fos = new FileOutputStream(targetFile);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             DataOutputStream dos = new DataOutputStream(bos)) {

            bos.write(header.array());

            // Konwersja float[-1.0, 1.0] -> 16-bit signed PCM (Little-Endian)
            byte[] buffer = new byte[Math.min(numSamples * 2, 4096)];
            int bufIdx = 0;

            for (int i = 0; i < numSamples; i++) {
                float sample = samples[i];
                // Clamping
                if (sample > 1.0f) sample = 1.0f;
                else if (sample < -1.0f) sample = -1.0f;

                short pcm = (short) Math.round(sample * 32767.0f);

                buffer[bufIdx++] = (byte) (pcm & 0xFF);
                buffer[bufIdx++] = (byte) ((pcm >> 8) & 0xFF);

                if (bufIdx >= buffer.length) {
                    bos.write(buffer, 0, bufIdx);
                    bufIdx = 0;
                }
            }

            if (bufIdx > 0) {
                bos.write(buffer, 0, bufIdx);
            }

            bos.flush();
        }
    }

    /**
     * Bezpiecznie usuwa pojedynczy plik audio na podstawie jego ścieżki.
     *
     * @param filePath bezwzględna ścieżka do pliku
     * @return true jeśli plik istniał i został pomyślnie usunięty
     */
    public static boolean deleteAudioFile(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            return false;
        }
        try {
            File f = new File(filePath);
            if (f.exists() && f.isFile()) {
                return f.delete();
            }
        } catch (Exception e) {
            Log.e(TAG, "Błąd podczas usuwania pliku audio: " + filePath, e);
        }
        return false;
    }

    /**
     * Czyści wszystkie zarejestrowane pliki WAV w katalogach nagrań aplikacji.
     *
     * @param context kontekst aplikacji Android
     * @return true jeśli proces zakończył się bez błędów
     */
    public static boolean deleteAllRecordings(Context context) {
        if (context == null) return false;
        boolean allSuccess = true;

        File[] dirs = new File[]{
            new File(context.getFilesDir(), RECORDINGS_SUBDIR),
            context.getExternalFilesDir(null) != null ? new File(context.getExternalFilesDir(null), RECORDINGS_SUBDIR) : null
        };

        for (File dir : dirs) {
            if (dir != null && dir.exists() && dir.isDirectory()) {
                File[] files = dir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        if (f.isFile() && f.getName().toLowerCase(Locale.US).endsWith(".wav")) {
                            if (!f.delete()) {
                                allSuccess = false;
                                Log.w(TAG, "Nie udało się usunąć nagrania: " + f.getAbsolutePath());
                            }
                        }
                    }
                }
            }
        }

        return allSuccess;
    }
}
