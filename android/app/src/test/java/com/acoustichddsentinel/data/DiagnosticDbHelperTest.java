package com.acoustichddsentinel.data;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe schematu bazy SQLite oraz bezpiecznej migracji v1 -> v2 w DiagnosticDbHelper.
 * Weryfikuje:
 * 1. Zgodność wersji bazy (DATABASE_VERSION = 2) i stałych kolumn.
 * 2. Spójność tablicy zapytań migracyjnych MIGRATION_V1_TO_V2_SQL (9 kolumn).
 * 3. Rzeczywistą migrację w silniku SQLite (sqlite3) z zachowaniem 100% danych użytkownika (Zero Data Loss).
 */
public class DiagnosticDbHelperTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testDatabaseVersionAndSchemaConstants() {
        assertEquals("acoustic_sentinel.db", DiagnosticDbHelper.DATABASE_NAME);
        assertEquals(2, DiagnosticDbHelper.DATABASE_VERSION);
        assertEquals("Diagnostics", DiagnosticDbHelper.TABLE_DIAGNOSTICS);

        // Kolumny bazowe (v1)
        assertEquals("id", DiagnosticDbHelper.COLUMN_ID);
        assertEquals("hdd_name", DiagnosticDbHelper.COLUMN_HDD_NAME);
        assertEquals("timestamp", DiagnosticDbHelper.COLUMN_TIMESTAMP);
        assertEquals("has_anomaly", DiagnosticDbHelper.COLUMN_HAS_ANOMALY);
        assertEquals("spectrogram_path", DiagnosticDbHelper.COLUMN_SPECTROGRAM_PATH);
        assertEquals("predicted_class", DiagnosticDbHelper.COLUMN_PREDICTED_CLASS);
        assertEquals("confidence", DiagnosticDbHelper.COLUMN_CONFIDENCE);

        // Nowe kolumny schematu (v2)
        assertEquals("audio_path", DiagnosticDbHelper.COLUMN_AUDIO_PATH);
        assertEquals("prob_beeping", DiagnosticDbHelper.COLUMN_PROB_BEEPING);
        assertEquals("prob_clicking", DiagnosticDbHelper.COLUMN_PROB_CLICKING);
        assertEquals("prob_normal", DiagnosticDbHelper.COLUMN_PROB_NORMAL);
        assertEquals("prob_scratching", DiagnosticDbHelper.COLUMN_PROB_SCRATCHING);
        assertEquals("crest_factor", DiagnosticDbHelper.COLUMN_CREST_FACTOR);
        assertEquals("peak_rms", DiagnosticDbHelper.COLUMN_PEAK_RMS);
        assertEquals("scan_duration_s", DiagnosticDbHelper.COLUMN_SCAN_DURATION_S);
        assertEquals("recommendation", DiagnosticDbHelper.COLUMN_RECOMMENDATION);
    }

    @Test
    public void testMigrationStatementsIntegrity() {
        assertNotNull("Tablica MIGRATION_V1_TO_V2_SQL nie może być null", DiagnosticDbHelper.MIGRATION_V1_TO_V2_SQL);
        assertEquals("Migracja v1 -> v2 musi definiować dokładnie 9 zapytań ALTER TABLE",
                9, DiagnosticDbHelper.MIGRATION_V1_TO_V2_SQL.length);

        List<String> expectedColumns = Arrays.asList(
            DiagnosticDbHelper.COLUMN_AUDIO_PATH,
            DiagnosticDbHelper.COLUMN_PROB_BEEPING,
            DiagnosticDbHelper.COLUMN_PROB_CLICKING,
            DiagnosticDbHelper.COLUMN_PROB_NORMAL,
            DiagnosticDbHelper.COLUMN_PROB_SCRATCHING,
            DiagnosticDbHelper.COLUMN_CREST_FACTOR,
            DiagnosticDbHelper.COLUMN_PEAK_RMS,
            DiagnosticDbHelper.COLUMN_SCAN_DURATION_S,
            DiagnosticDbHelper.COLUMN_RECOMMENDATION
        );

        for (int i = 0; i < DiagnosticDbHelper.MIGRATION_V1_TO_V2_SQL.length; i++) {
            String sql = DiagnosticDbHelper.MIGRATION_V1_TO_V2_SQL[i];
            assertNotNull("Zapytanie migracyjne #" + i + " nie może być null", sql);
            assertTrue("Zapytanie migracyjne #" + i + " musi zawierać ALTER TABLE Diagnostics ADD COLUMN",
                    sql.startsWith("ALTER TABLE " + DiagnosticDbHelper.TABLE_DIAGNOSTICS + " ADD COLUMN "));

            String expectedCol = expectedColumns.get(i);
            assertTrue("Zapytanie #" + i + " musi dodawać kolumnę: " + expectedCol,
                    sql.contains(" " + expectedCol + " "));
        }

        // Weryfikacja definicji tabeli v1 (brak nowych kolumn)
        String v1Create = DiagnosticDbHelper.SQL_CREATE_TABLE_DIAGNOSTICS_V1;
        assertTrue(v1Create.contains(DiagnosticDbHelper.COLUMN_HDD_NAME));
        assertTrue(v1Create.contains(DiagnosticDbHelper.COLUMN_CONFIDENCE));
        assertFalse(v1Create.contains(DiagnosticDbHelper.COLUMN_AUDIO_PATH));
        assertFalse(v1Create.contains(DiagnosticDbHelper.COLUMN_CREST_FACTOR));

        // Weryfikacja definicji tabeli v2 (komplet 16 kolumn)
        String v2Create = DiagnosticDbHelper.SQL_CREATE_TABLE_DIAGNOSTICS;
        assertTrue(v2Create.contains(DiagnosticDbHelper.COLUMN_AUDIO_PATH));
        assertTrue(v2Create.contains(DiagnosticDbHelper.COLUMN_CREST_FACTOR));
        assertTrue(v2Create.contains(DiagnosticDbHelper.COLUMN_RECOMMENDATION));
    }

    @Test
    public void testSafeSqliteMigrationV1ToV2WithoutDataLoss() throws Exception {
        // Sprawdź czy środowisko uruchomieniowe posiada zainstalowany silnik sqlite3
        File sqliteBin = new File("/usr/bin/sqlite3");
        if (!sqliteBin.exists() && !sqliteBin.canExecute()) {
            System.out.println("sqlite3 CLI nie jest dostępne — pomijam test integracyjny procesu.");
            return;
        }

        File dbFile = tempFolder.newFile("test_migration.db");

        // 1. Inicjalizacja bazy w schemacie v1
        executeSqlite(dbFile, DiagnosticDbHelper.SQL_CREATE_TABLE_DIAGNOSTICS_V1);

        // 2. Wstawienie wpisów w schemacie v1
        executeSqlite(dbFile,
            "INSERT INTO Diagnostics (hdd_name, timestamp, has_anomaly, spectrogram_path, predicted_class, confidence) " +
            "VALUES ('Seagate ST2000DM001', 1731376800000, 1, '/data/spec_1.png', 'clicking', 0.942);"
        );
        executeSqlite(dbFile,
            "INSERT INTO Diagnostics (hdd_name, timestamp, has_anomaly, spectrogram_path, predicted_class, confidence) " +
            "VALUES ('WD Blue 1TB', 1731380000000, 0, '/data/spec_2.png', 'normal', 0.991);"
        );

        // 3. Wykonanie kroków migracji v1 -> v2 (dokładnie takich samych jak w onUpgrade)
        for (String migrationSql : DiagnosticDbHelper.MIGRATION_V1_TO_V2_SQL) {
            executeSqlite(dbFile, migrationSql);
        }

        // 4. Weryfikacja liczby kolumn po migracji (musi być 16)
        List<String> pragmaCols = querySqlite(dbFile, "PRAGMA table_info(Diagnostics);");
        assertEquals("Po migracji tabela musi mieć dokładnie 16 kolumn", 16, pragmaCols.size());

        // 5. Weryfikacja nienaruszalności dotychczasowych danych użytkownika (Zero Data Loss)
        List<String> rows = querySqlite(dbFile,
            "SELECT id, hdd_name, timestamp, has_anomaly, spectrogram_path, predicted_class, confidence, " +
            "audio_path, prob_beeping, prob_clicking, prob_normal, prob_scratching, crest_factor, peak_rms, scan_duration_s, recommendation " +
            "FROM Diagnostics ORDER BY id ASC;"
        );

        assertEquals("Baza musi nadal zawierać oba rekordy po migracji", 2, rows.size());

        String[] row1 = rows.get(0).split("\\|", -1);
        assertEquals("1", row1[0]); // id
        assertEquals("Seagate ST2000DM001", row1[1]); // hdd_name
        assertEquals("1731376800000", row1[2]); // timestamp
        assertEquals("1", row1[3]); // has_anomaly
        assertEquals("/data/spec_1.png", row1[4]); // spectrogram_path
        assertEquals("clicking", row1[5]); // predicted_class
        assertEquals("0.942", row1[6]); // confidence
        assertEquals("", row1[7]); // audio_path domyślnie NULL (wypisane jako pusty ciąg)
        assertEquals("0.0", row1[8]); // prob_beeping DEFAULT 0.0
        assertEquals("0.0", row1[9]); // prob_clicking DEFAULT 0.0
        assertEquals("0.0", row1[10]); // prob_normal DEFAULT 0.0
        assertEquals("0.0", row1[11]); // prob_scratching DEFAULT 0.0
        assertEquals("0.0", row1[12]); // crest_factor DEFAULT 0.0
        assertEquals("0.0", row1[13]); // peak_rms DEFAULT 0.0
        assertEquals("0.0", row1[14]); // scan_duration_s DEFAULT 0.0
        assertEquals("", row1[15]); // recommendation domyślnie NULL

        String[] row2 = rows.get(1).split("\\|", -1);
        assertEquals("2", row2[0]);
        assertEquals("WD Blue 1TB", row2[1]);
        assertEquals("0", row2[3]); // has_anomaly == 0
        assertEquals("normal", row2[5]);

        // 6. Weryfikacja możliwości zapisu nowego rekordu w pełnym schemacie v2
        executeSqlite(dbFile,
            "INSERT INTO Diagnostics (" +
            "hdd_name, timestamp, has_anomaly, spectrogram_path, predicted_class, confidence, " +
            "audio_path, prob_beeping, prob_clicking, prob_normal, prob_scratching, crest_factor, peak_rms, scan_duration_s, recommendation" +
            ") VALUES (" +
            "'Toshiba P300 3TB', 1731385000000, 1, '/data/spec_3.png', 'beeping', 0.885, " +
            "'/data/rec_beeping.wav', 0.885, 0.02, 0.01, 0.085, 4.25, 0.120, 3.0, 'Zatarty silnik wrzeciona'" +
            ");"
        );

        List<String> updatedRows = querySqlite(dbFile, "SELECT COUNT(*) FROM Diagnostics;");
        assertEquals("Po wstawieniu rekordu v2 tabela powinna mieć 3 wiersze", "3", updatedRows.get(0).trim());
    }

    private static void executeSqlite(File dbFile, String sql) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("sqlite3", dbFile.getAbsolutePath(), sql);
        pb.redirectErrorStream(true);
        Process process = pb.start();
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append("\n");
                }
                throw new RuntimeException("Błąd wykonania SQLite: " + sb.toString());
            }
        }
    }

    private static List<String> querySqlite(File dbFile, String sql) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("sqlite3", dbFile.getAbsolutePath(), sql);
        Process process = pb.start();
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Błąd zapytania SQLite (kod " + exitCode + ")");
        }
        return lines;
    }
}
