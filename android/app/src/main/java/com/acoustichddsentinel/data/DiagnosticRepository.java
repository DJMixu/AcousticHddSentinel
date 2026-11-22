package com.acoustichddsentinel.data;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.os.Handler;
import android.os.Looper;

import com.acoustichddsentinel.audio.WavAudioSaver;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Repozytorium trwałych danych diagnostycznych (Repository Pattern).
 * Zapewnia asynchroniczną i bezpieczną wielowątkowo obsługę bazy SQLite z powiadamianiem wątku głównego UI.
 * Obsługuje wzbogacony model diagnostyczny (schemat v2) wraz z czyszczeniem powiązanych plików
 * spektrogramów PNG oraz archiwalnych nagrań audio WAV.
 */
public class DiagnosticRepository {

    public interface RepositoryCallback<T> {
        void onComplete(T result);
    }

    private final Context context;
    private final DiagnosticDbHelper dbHelper;
    private final ExecutorService executorService;
    private final Handler mainThreadHandler;

    public DiagnosticRepository(Context context) {
        this.context = context.getApplicationContext();
        this.dbHelper = new DiagnosticDbHelper(this.context);
        this.executorService = Executors.newSingleThreadExecutor();
        this.mainThreadHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * Wstawia nowy wynik diagnozy do bazy danych z zapisem wszystkich parametrów telemetrycznych.
     */
    public void insert(DiagnosticModel model, RepositoryCallback<Long> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();
            long newId = dbHelper.insertDiagnostic(db, model);
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
            SQLiteDatabase db = dbHelper.getReadableDatabase();
            List<DiagnosticModel> list = dbHelper.getAllDiagnostics(db);
            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(list));
            }
        });
    }

    /**
     * Usuwa pojedynczy wpis z bazy oraz skojarzony plik spektrogramu PNG i nagranie audio WAV.
     */
    public void deleteById(long id, RepositoryCallback<Boolean> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();

            String specPath = null;
            String audioPath = null;
            try (Cursor c = db.query(
                    DiagnosticDbHelper.TABLE_DIAGNOSTICS,
                    new String[]{DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH, DiagnosticDbHelper.COLUMN_AUDIO_PATH},
                    DiagnosticDbHelper.COLUMN_ID + " = ?",
                    new String[]{String.valueOf(id)},
                    null, null, null)) {
                if (c.moveToFirst()) {
                    specPath = c.getString(0);
                    if (!c.isNull(1)) {
                        audioPath = c.getString(1);
                    }
                }
            }

            // Usunięcie powiązanego pliku spektrogramu PNG
            if (specPath != null && !specPath.isEmpty()) {
                File imgFile = new File(specPath);
                if (imgFile.exists()) {
                    imgFile.delete();
                }
            }

            // Usunięcie powiązanego nagrania audio WAV
            if (audioPath != null && !audioPath.isEmpty()) {
                WavAudioSaver.deleteAudioFile(audioPath);
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
     * Czyści całą historię badań i usuwa pliki spektrogramów oraz nagrań audio.
     */
    public void clearAll(RepositoryCallback<Boolean> callback) {
        executorService.execute(() -> {
            SQLiteDatabase db = dbHelper.getWritableDatabase();

            try (Cursor c = db.query(
                    DiagnosticDbHelper.TABLE_DIAGNOSTICS,
                    new String[]{DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH, DiagnosticDbHelper.COLUMN_AUDIO_PATH},
                    null, null, null, null, null)) {
                while (c.moveToNext()) {
                    String specPath = c.getString(0);
                    if (specPath != null && !specPath.isEmpty()) {
                        File imgFile = new File(specPath);
                        if (imgFile.exists()) {
                            imgFile.delete();
                        }
                    }
                    if (!c.isNull(1)) {
                        String audioPath = c.getString(1);
                        if (audioPath != null && !audioPath.isEmpty()) {
                            WavAudioSaver.deleteAudioFile(audioPath);
                        }
                    }
                }
            }

            // Dodatkowe wyczyszczenie ewentualnych osieroconych plików nagrań
            WavAudioSaver.deleteAllRecordings(context);

            int deletedRows = db.delete(DiagnosticDbHelper.TABLE_DIAGNOSTICS, null, null);
            boolean success = deletedRows >= 0;
            if (callback != null) {
                mainThreadHandler.post(() -> callback.onComplete(success));
            }
        });
    }
}
