package com.acoustichddsentinel.data;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

/**
 * Natywny zarządca bazy danych SQLite dla platformy Android (SQLiteOpenHelper).
 * Spełnia wymagania formalne sylabusa laboratoryjnego (Projekt_AM.pdf) bez narzutu zewnętrznych bibliotek ORM.
 */
public class DiagnosticDbHelper extends SQLiteOpenHelper {

    public static final String DATABASE_NAME = "acoustic_sentinel.db";
    public static final int DATABASE_VERSION = 1;

    public static final String TABLE_DIAGNOSTICS = "Diagnostics";
    public static final String COLUMN_ID = "id";
    public static final String COLUMN_HDD_NAME = "hdd_name";
    public static final String COLUMN_TIMESTAMP = "timestamp";
    public static final String COLUMN_HAS_ANOMALY = "has_anomaly";
    public static final String COLUMN_SPECTROGRAM_PATH = "spectrogram_path";
    public static final String COLUMN_PREDICTED_CLASS = "predicted_class";
    public static final String COLUMN_CONFIDENCE = "confidence";

    private static final String SQL_CREATE_TABLE_DIAGNOSTICS =
        "CREATE TABLE " + TABLE_DIAGNOSTICS + " (" +
        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
        COLUMN_HDD_NAME + " TEXT NOT NULL, " +
        COLUMN_TIMESTAMP + " INTEGER NOT NULL, " +
        COLUMN_HAS_ANOMALY + " INTEGER NOT NULL, " +
        COLUMN_SPECTROGRAM_PATH + " TEXT, " +
        COLUMN_PREDICTED_CLASS + " TEXT NOT NULL, " +
        COLUMN_CONFIDENCE + " REAL NOT NULL" +
        ");";

    private static final String SQL_DROP_TABLE_DIAGNOSTICS =
        "DROP TABLE IF EXISTS " + TABLE_DIAGNOSTICS;

    public DiagnosticDbHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(SQL_CREATE_TABLE_DIAGNOSTICS);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL(SQL_DROP_TABLE_DIAGNOSTICS);
        onCreate(db);
    }
}
