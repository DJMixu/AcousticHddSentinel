package com.acoustichddsentinel.data;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Handler;
import android.os.Looper;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Repozytorium trwałych danych diagnostycznych (Repository Pattern).
 * Zapewnia asynchroniczną i bezpieczną wielowątkowo obsługę bazy SQLite z powiadamianiem wątku głównego UI.
 */
public class DiagnosticRepository {

    public interface RepositoryCallback<T> {
        void onComplete(T result);
    }

    private final DiagnosticDbHelper dbHelper;
    private final ExecutorService executorService;
    private final Handler mainThreadHandler;

    public DiagnosticRepository(Context context) {
        this.dbHelper = new DiagnosticDbHelper(context.getApplicationContext());
        this.executorService = Executors.newSingleThreadExecutor();
        this.mainThreadHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * Wstawia nowy wynik diagnozy do bazy danych.
     */
    public void insert(DiagnosticModel model, RepositoryCallback<Long> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();
            ContentValues values = new ContentValues();
            values.put(DiagnosticDbHelper.COLUMN_HDD_NAME, model.getHddName());
            values.put(DiagnosticDbHelper.COLUMN_TIMESTAMP, model.getTimestamp());
            values.put(DiagnosticDbHelper.COLUMN_HAS_ANOMALY, model.hasAnomaly() ? 1 : 0);
            values.put(DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH, model.getSpectrogramPath());
            values.put(DiagnosticDbHelper.COLUMN_PREDICTED_CLASS, model.getPredictedClass());
            values.put(DiagnosticDbHelper.COLUMN_CONFIDENCE, model.getConfidence());

            long newId = db.insert(DiagnosticDbHelper.TABLE_DIAGNOSTICS, null, values);
            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(newId));
            }
        });
    }

    /**
     * Pobiera całą historię badań posortowaną od najnowszego do najstarszego.
     */
    public void getAll(RepositoryCallback<List<DiagnosticModel>> callback) {
        executorService.execute(() -> {
            List<DiagnosticModel> list = new ArrayList<>();
            SQLiteDatabase db = dbHelper.getReadableDatabase();
            String query = "SELECT * FROM " + DiagnosticDbHelper.TABLE_DIAGNOSTICS +
                           " ORDER BY " + DiagnosticDbHelper.COLUMN_TIMESTAMP + " DESC";

            try (Cursor cursor = db.rawQuery(query, null)) {
                if (cursor.moveToFirst()) {
                    int idCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_ID);
                    int hddCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_HDD_NAME);
                    int tsCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_TIMESTAMP);
                    int anomalyCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_HAS_ANOMALY);
                    int specCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH);
                    int classCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_PREDICTED_CLASS);
                    int confCol = cursor.getColumnIndexOrThrow(DiagnosticDbHelper.COLUMN_CONFIDENCE);

                    do {
                        long id = cursor.getLong(idCol);
                        String hddName = cursor.getString(hddCol);
                        long ts = cursor.getLong(tsCol);
                        boolean hasAnomaly = cursor.getInt(anomalyCol) == 1;
                        String specPath = cursor.getString(specCol);
                        String predClass = cursor.getString(classCol);
                        float conf = cursor.getFloat(confCol);

                        list.add(new DiagnosticModel(id, hddName, ts, hasAnomaly, specPath, predClass, conf));
                    } while (cursor.moveToNext());
                }
            }

            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(list));
            }
        });
    }

    /**
     * Usuwa pojedynczy wpis z bazy oraz skojarzony plik spektrogramu PNG z pamięci wewnętrznej.
     */
    public void deleteById(long id, RepositoryCallback<Boolean> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();

            // Pobierz ścieżkę pliku obrazu, aby usunąć plik PNG
            String path = null;
            try (Cursor c = db.query(
                    DiagnosticDbHelper.TABLE_DIAGNOSTICS,
                    new String[]{DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH},
                    DiagnosticDbHelper.COLUMN_ID + " = ?",
                    new String[]{String.valueOf(id)},
                    null, null, null)) {
                if (c.moveToFirst()) {
                    path = c.getString(0);
                }
            }

            if (path != null && !path.isEmpty()) {
                File imgFile = new File(path);
                if (imgFile.exists()) {
                    imgFile.delete();
                }
            }

            int deletedRows = db.delete(
                DiagnosticDbHelper.TABLE_DIAGNOSTICS,
                DiagnosticDbHelper.COLUMN_ID + " = ?",
                new String[]{String.valueOf(id)}
            );

            boolean success = deletedRows > 0;
            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(success));
            }
        });
    }

    /**
     * Czyści całą historię badań i usuwa pliki spektrogramów.
     */
    public void clearAll(RepositoryCallback<Boolean> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();
            // Odczyt wszystkich ścieżek
            try (Cursor c = db.query(
                    DiagnosticDbHelper.TABLE_DIAGNOSTICS,
                    new String[]{DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH},
                    null, null, null, null, null)) {
                while (c.moveToNext()) {
                    String path = c.getString(0);
                    if (path != null && !path.isEmpty()) {
                        File imgFile = new File(path);
                        if (imgFile.exists()) {
                            imgFile.delete();
                        }
                    }
                }
            }

            int deletedRows = db.delete(DiagnosticDbHelper.TABLE_DIAGNOSTICS, null, null);
            boolean success = deletedRows >= 0;
            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(success));
            }
        });
    }
}
