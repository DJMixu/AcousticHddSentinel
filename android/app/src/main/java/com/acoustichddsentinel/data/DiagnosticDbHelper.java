package com.acoustichddsentinel.data;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Natywny zarządca bazy danych SQLite dla platformy Android (SQLiteOpenHelper).
 * Spełnia wymagania formalne sylabusa laboratoryjnego (Projekt_AM.pdf) bez narzutu zewnętrznych bibliotek ORM.
 * Wersja schematu 2 wzbogaca rejestr o ścieżki nagrań WAV, rozkłady prawdopodobieństw klas,
 * parametry Crest Factor, Peak RMS, czas trwania oraz rekomendacje naprawcze.
 */
public class DiagnosticDbHelper extends SQLiteOpenHelper {

    private static final String TAG = "DiagnosticDbHelper";

    public static final String DATABASE_NAME = "acoustic_sentinel.db";
    public static final int DATABASE_VERSION = 2;

    public static final String TABLE_DIAGNOSTICS = "Diagnostics";
    public static final String COLUMN_ID = "id";
    public static final String COLUMN_HDD_NAME = "hdd_name";
    public static final String COLUMN_TIMESTAMP = "timestamp";
    public static final String COLUMN_HAS_ANOMALY = "has_anomaly";
    public static final String COLUMN_SPECTROGRAM_PATH = "spectrogram_path";
    public static final String COLUMN_PREDICTED_CLASS = "predicted_class";
    public static final String COLUMN_CONFIDENCE = "confidence";

    // Nowe kolumny w schemacie v2:
    public static final String COLUMN_AUDIO_PATH = "audio_path";
    public static final String COLUMN_PROB_BEEPING = "prob_beeping";
    public static final String COLUMN_PROB_CLICKING = "prob_clicking";
    public static final String COLUMN_PROB_NORMAL = "prob_normal";
    public static final String COLUMN_PROB_SCRATCHING = "prob_scratching";
    public static final String COLUMN_CREST_FACTOR = "crest_factor";
    public static final String COLUMN_PEAK_RMS = "peak_rms";
    public static final String COLUMN_SCAN_DURATION_S = "scan_duration_s";
    public static final String COLUMN_RECOMMENDATION = "recommendation";

    public static final String SQL_CREATE_TABLE_DIAGNOSTICS_V1 =
        "CREATE TABLE " + TABLE_DIAGNOSTICS + " (" +
        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
        COLUMN_HDD_NAME + " TEXT NOT NULL, " +
        COLUMN_TIMESTAMP + " INTEGER NOT NULL, " +
        COLUMN_HAS_ANOMALY + " INTEGER NOT NULL, " +
        COLUMN_SPECTROGRAM_PATH + " TEXT, " +
        COLUMN_PREDICTED_CLASS + " TEXT NOT NULL, " +
        COLUMN_CONFIDENCE + " REAL NOT NULL" +
        ");";

    public static final String[] MIGRATION_V1_TO_V2_SQL = new String[]{
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_AUDIO_PATH + " TEXT;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_PROB_BEEPING + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_PROB_CLICKING + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_PROB_NORMAL + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_PROB_SCRATCHING + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_CREST_FACTOR + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_PEAK_RMS + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_SCAN_DURATION_S + " REAL DEFAULT 0.0;",
        "ALTER TABLE " + TABLE_DIAGNOSTICS + " ADD COLUMN " + COLUMN_RECOMMENDATION + " TEXT;"
    };

    public static final String SQL_CREATE_TABLE_DIAGNOSTICS =
        "CREATE TABLE " + TABLE_DIAGNOSTICS + " (" +
        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
        COLUMN_HDD_NAME + " TEXT NOT NULL, " +
        COLUMN_TIMESTAMP + " INTEGER NOT NULL, " +
        COLUMN_HAS_ANOMALY + " INTEGER NOT NULL, " +
        COLUMN_SPECTROGRAM_PATH + " TEXT, " +
        COLUMN_PREDICTED_CLASS + " TEXT NOT NULL, " +
        COLUMN_CONFIDENCE + " REAL NOT NULL, " +
        COLUMN_AUDIO_PATH + " TEXT, " +
        COLUMN_PROB_BEEPING + " REAL DEFAULT 0.0, " +
        COLUMN_PROB_CLICKING + " REAL DEFAULT 0.0, " +
        COLUMN_PROB_NORMAL + " REAL DEFAULT 0.0, " +
        COLUMN_PROB_SCRATCHING + " REAL DEFAULT 0.0, " +
        COLUMN_CREST_FACTOR + " REAL DEFAULT 0.0, " +
        COLUMN_PEAK_RMS + " REAL DEFAULT 0.0, " +
        COLUMN_SCAN_DURATION_S + " REAL DEFAULT 0.0, " +
        COLUMN_RECOMMENDATION + " TEXT" +
        ");";

    public DiagnosticDbHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(SQL_CREATE_TABLE_DIAGNOSTICS);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        Log.i(TAG, "Migracja bazy danych z wersji " + oldVersion + " do " + newVersion);
        if (oldVersion < 2) {
            // Bezpieczna migracja v1 -> v2 z zachowaniem dotychczasowych wpisów użytkownika
            try {
                for (String sql : MIGRATION_V1_TO_V2_SQL) {
                    db.execSQL(sql);
                }
                Log.i(TAG, "Pomyślnie zaktualizowano schemat bazy do v2 (dodano 9 kolumn).");
            } catch (Exception e) {
                Log.e(TAG, "Błąd podczas alter table, fallback do odtworzenia tabeli", e);
                db.execSQL("DROP TABLE IF EXISTS " + TABLE_DIAGNOSTICS);
                onCreate(db);
            }
        }
    }

    /**
     * Wstawia nowy wpis diagnostyczny do przekazanej instancji bazy danych.
     */
    public long insertDiagnostic(SQLiteDatabase db, DiagnosticModel model) {
        ContentValues values = new ContentValues();
        values.put(COLUMN_HDD_NAME, model.getHddName());
        values.put(COLUMN_TIMESTAMP, model.getTimestamp());
        values.put(COLUMN_HAS_ANOMALY, model.hasAnomaly() ? 1 : 0);
        values.put(COLUMN_SPECTROGRAM_PATH, model.getSpectrogramPath());
        values.put(COLUMN_PREDICTED_CLASS, model.getPredictedClass());
        values.put(COLUMN_CONFIDENCE, model.getConfidence());
        values.put(COLUMN_AUDIO_PATH, model.getAudioPath());
        values.put(COLUMN_PROB_BEEPING, model.getProbBeeping());
        values.put(COLUMN_PROB_CLICKING, model.getProbClicking());
        values.put(COLUMN_PROB_NORMAL, model.getProbNormal());
        values.put(COLUMN_PROB_SCRATCHING, model.getProbScratching());
        values.put(COLUMN_CREST_FACTOR, model.getCrestFactor());
        values.put(COLUMN_PEAK_RMS, model.getPeakRms());
        values.put(COLUMN_SCAN_DURATION_S, model.getScanDurationS());
        values.put(COLUMN_RECOMMENDATION, model.getRecommendation());

        return db.insert(TABLE_DIAGNOSTICS, null, values);
    }

    /**
     * Przeciążenie dla zapisu z użyciem wewnętrznego połączenia do zapisu.
     */
    public long insertDiagnostic(DiagnosticModel model) {
        SQLiteDatabase db = getWritableDatabase();
        return insertDiagnostic(db, model);
    }

    /**
     * Pobiera wszystkie rekordy diagnostyczne z bazy danych posortowane malejąco po znaczniku czasu.
     */
    public List<DiagnosticModel> getAllDiagnostics(SQLiteDatabase db) {
        List<DiagnosticModel> list = new ArrayList<>();
        String query = "SELECT * FROM " + TABLE_DIAGNOSTICS +
                       " ORDER BY " + COLUMN_TIMESTAMP + " DESC";

        try (Cursor cursor = db.rawQuery(query, null)) {
            if (cursor.moveToFirst()) {
                int idCol = cursor.getColumnIndexOrThrow(COLUMN_ID);
                int hddCol = cursor.getColumnIndexOrThrow(COLUMN_HDD_NAME);
                int tsCol = cursor.getColumnIndexOrThrow(COLUMN_TIMESTAMP);
                int anomalyCol = cursor.getColumnIndexOrThrow(COLUMN_HAS_ANOMALY);
                int specCol = cursor.getColumnIndexOrThrow(COLUMN_SPECTROGRAM_PATH);
                int classCol = cursor.getColumnIndexOrThrow(COLUMN_PREDICTED_CLASS);
                int confCol = cursor.getColumnIndexOrThrow(COLUMN_CONFIDENCE);

                int audioCol = cursor.getColumnIndex(COLUMN_AUDIO_PATH);
                int pBeepCol = cursor.getColumnIndex(COLUMN_PROB_BEEPING);
                int pClickCol = cursor.getColumnIndex(COLUMN_PROB_CLICKING);
                int pNormCol = cursor.getColumnIndex(COLUMN_PROB_NORMAL);
                int pScratchCol = cursor.getColumnIndex(COLUMN_PROB_SCRATCHING);
                int crestCol = cursor.getColumnIndex(COLUMN_CREST_FACTOR);
                int peakRmsCol = cursor.getColumnIndex(COLUMN_PEAK_RMS);
                int durCol = cursor.getColumnIndex(COLUMN_SCAN_DURATION_S);
                int recCol = cursor.getColumnIndex(COLUMN_RECOMMENDATION);

                do {
                    long id = cursor.getLong(idCol);
                    String hddName = cursor.getString(hddCol);
                    long ts = cursor.getLong(tsCol);
                    boolean hasAnomaly = cursor.getInt(anomalyCol) == 1;
                    String specPath = cursor.getString(specCol);
                    String predClass = cursor.getString(classCol);
                    float conf = cursor.getFloat(confCol);

                    String audioPath = (audioCol != -1 && !cursor.isNull(audioCol)) ? cursor.getString(audioCol) : null;
                    float probBeeping = (pBeepCol != -1 && !cursor.isNull(pBeepCol)) ? cursor.getFloat(pBeepCol) : 0.0f;
                    float probClicking = (pClickCol != -1 && !cursor.isNull(pClickCol)) ? cursor.getFloat(pClickCol) : 0.0f;
                    float probNormal = (pNormCol != -1 && !cursor.isNull(pNormCol)) ? cursor.getFloat(pNormCol) : 0.0f;
                    float probScratching = (pScratchCol != -1 && !cursor.isNull(pScratchCol)) ? cursor.getFloat(pScratchCol) : 0.0f;
                    float crestFactor = (crestCol != -1 && !cursor.isNull(crestCol)) ? cursor.getFloat(crestCol) : 0.0f;
                    float peakRms = (peakRmsCol != -1 && !cursor.isNull(peakRmsCol)) ? cursor.getFloat(peakRmsCol) : 0.0f;
                    float scanDurationS = (durCol != -1 && !cursor.isNull(durCol)) ? cursor.getFloat(durCol) : 3.0f;
                    String recommendation = (recCol != -1 && !cursor.isNull(recCol)) ? cursor.getString(recCol) : null;

                    list.add(new DiagnosticModel(
                        id, hddName, ts, hasAnomaly, specPath, predClass, conf,
                        audioPath, probBeeping, probClicking, probNormal, probScratching,
                        crestFactor, peakRms, scanDurationS, recommendation
                    ));
                } while (cursor.moveToNext());
            }
        }
        return list;
    }

    /**
     * Przeciążenie dla pobierania z użyciem wewnętrznego połączenia do odczytu.
     */
    public List<DiagnosticModel> getAllDiagnostics() {
        SQLiteDatabase db = getReadableDatabase();
        return getAllDiagnostics(db);
    }
}
