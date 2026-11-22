package com.acoustichddsentinel.dsp;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Konwerter spektrogramu Mel do obrazu graficznego Bitmap i pliku PNG.
 * Zapisuje wygenerowane obrazy w pamięci wewnętrznej aplikacji (Internal Storage)
 * oraz umożliwia eksport bezpośrednio do publicznej Galerii zdjęć (Pictures/AcousticHddSentinel).
 * Zgodność z ADR-002 i NFR-04.
 */
public final class SpectrogramImageSaver {

    private static final String TAG = "SpectrogramImageSaver";
    private static final String SPECTROGRAMS_SUBDIR = "spectrograms";
    public static final String GALLERY_SUBDIR = "Pictures/AcousticHddSentinel";

    private SpectrogramImageSaver() {
    }

    /**
     * Konwertuje macierz Log-Mel [128][259] na kolorową mapę cieplną Bitmap (Inferno/Plasma).
     * Oś Y jest odwracana (niskie częstotliwości na dole, wysokie na górze).
     *
     * @param logMel macierz energii w skali dB (zakres ok. [-80.0, 0.0])
     * @return obiekt Bitmap w formacie ARGB_8888 o wymiarach 259 x 128
     */
    public static Bitmap createSpectrogramBitmap(float[][] logMel) {
        int numMels = logMel.length;       // 128
        int timeSteps = logMel[0].length;  // 259

        int width = timeSteps;
        int height = numMels;
        int[] pixels = new int[width * height];

        for (int m = 0; m < numMels; m++) {
            // Niskie częstotliwości (m = 0) umieszczamy na dole obrazu (y = height - 1)
            int y = (height - 1) - m;
            for (int t = 0; t < timeSteps; t++) {
                float db = logMel[m][t];
                // Normalizacja dB [-80.0, 0.0] do [0.0, 1.0]
                float norm = (db + DspConstants.TOP_DB) / DspConstants.TOP_DB;
                if (norm < 0.0f) norm = 0.0f;
                if (norm > 1.0f) norm = 1.0f;

                int color = interpolateInfernoColor(norm);
                pixels[y * width + t] = color;
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    /**
     * Zapisuje wygenerowaną mapę spektrogramu do pliku PNG w katalogu prywatnym aplikacji.
     *
     * @param context kontekst aplikacji Android
     * @param bitmap  obiekt Bitmap spektrogramu
     * @param prefix  prefiks nazwy pliku (np. "spec_")
     * @return bezwzględna ścieżka do zapisanego pliku PNG na dysku
     */
    public static String saveSpectrogramPng(Context context, Bitmap bitmap, String prefix) {
        File dir = new File(context.getFilesDir(), SPECTROGRAMS_SUBDIR);
        if (!dir.exists() && !dir.mkdirs()) {
            Log.e(TAG, "Nie udało się utworzyć katalogu spektrogramów: " + dir.getAbsolutePath());
        }

        String fileName = prefix + "_" + System.currentTimeMillis() + ".png";
        File file = new File(dir, fileName);

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "Błąd podczas zapisu spektrogramu PNG do pliku", e);
            return null;
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    /**
     * Zapisuje wygenerowany spektrogram bezpośrednio do publicznej Galerii zdjęć urządzenia
     * (Pictures/AcousticHddSentinel) za pomocą interfejsu MediaStore Androida.
     *
     * @param context kontekst aplikacji Android
     * @param bitmap  obiekt Bitmap spektrogramu
     * @param title   tytuł lub prefiks nazwy pliku
     * @return Uri zapisanego obrazu w pamięci publicznej lub null przy błędzie
     */
    public static Uri saveSpectrogramToGallery(Context context, Bitmap bitmap, String title) {
        if (context == null || bitmap == null) return null;
        try {
            ContentResolver resolver = context.getContentResolver();
            String cleanTitle = (title != null && !title.trim().isEmpty())
                    ? title.replaceAll("[^a-zA-Z0-9_-]", "_")
                    : "spectrogram";
            String fileName = cleanTitle + "_" + System.currentTimeMillis() + ".png";

            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/AcousticHddSentinel");
                values.put(MediaStore.Images.Media.IS_PENDING, 1);
            }

            Uri collection = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                    ? MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                    : MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

            Uri imageUri = resolver.insert(collection, values);
            if (imageUri != null) {
                try (OutputStream out = resolver.openOutputStream(imageUri)) {
                    if (out != null) {
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                        out.flush();
                    }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear();
                    values.put(MediaStore.Images.Media.IS_PENDING, 0);
                    resolver.update(imageUri, values, null, null);
                }
                Log.i(TAG, "Spektrogram pomyślnie wyeksportowany do Galerii: " + imageUri);
                return imageUri;
            }
        } catch (Throwable t) {
            Log.e(TAG, "Nie udało się zapisać spektrogramu do Galerii MediaStore: " + t.getMessage(), t);
        }
        return null;
    }

    /**
     * Kopiuje istniejący plik spektrogramu z pamięci prywatnej do publicznej Galerii zdjęć.
     *
     * @param context      kontekst aplikacji Android
     * @param internalPath ścieżka bezwzględna do pliku PNG w pamięci aplikacji
     * @param title        tytuł lub nazwa badania
     * @return Uri zapisanego obrazu w Galerii lub null przy błędzie
     */
    public static Uri exportSpectrogramToGalleryFromPath(Context context, String internalPath, String title) {
        if (context == null || internalPath == null || internalPath.trim().isEmpty()) return null;
        File file = new File(internalPath);
        if (!file.exists() || !file.canRead()) {
            Log.w(TAG, "Plik spektrogramu nie istnieje lub brak uprawnień odczytu: " + internalPath);
            return null;
        }
        try {
            Bitmap bitmap = BitmapFactory.decodeFile(internalPath);
            if (bitmap != null) {
                return saveSpectrogramToGallery(context, bitmap, title);
            }
        } catch (Throwable t) {
            Log.e(TAG, "Błąd podczas dekodowania i eksportu spektrogramu z pliku: " + t.getMessage());
        }
        return null;
    }

    /**
     * Interpolacja palety termowizyjnej Inferno (od ciemnofioletowego, przez purpurę i pomarańcz, do żółci).
     */
    private static int interpolateInfernoColor(float t) {
        int r, g, b;
        if (t < 0.25f) {
            float s = t / 0.25f;
            r = (int) (10 + s * 60);
            g = (int) (5 + s * 15);
            b = (int) (30 + s * 80);
        } else if (t < 0.5f) {
            float s = (t - 0.25f) / 0.25f;
            r = (int) (70 + s * 110);
            g = (int) (20 + s * 30);
            b = (int) (110 - s * 30);
        } else if (t < 0.75f) {
            float s = (t - 0.5f) / 0.25f;
            r = (int) (180 + s * 60);
            g = (int) (50 + s * 100);
            b = (int) (80 - s * 50);
        } else {
            float s = (t - 0.75f) / 0.25f;
            r = (int) (240 + s * 15);
            g = (int) (150 + s * 105);
            b = (int) (30 + s * 170);
        }

        r = Math.min(255, Math.max(0, r));
        g = Math.min(255, Math.max(0, g));
        b = Math.min(255, Math.max(0, b));

        return Color.rgb(r, g, b);
    }
}
