package com.acoustichddsentinel.audio;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.SharedPreferences;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Zestaw testów jednostkowych preferencji toru audio (AudioPreferences)
 * oraz konfiguracji trybu czystego sygnału RAW w AudioRecorder (czysta Java 17, Zero Kotlin).
 */
public class AudioPreferencesTest {

    private MockSharedPreferences mockSharedPreferences;
    private MockContext mockContext;

    @Before
    public void setUp() {
        mockSharedPreferences = new MockSharedPreferences();
        mockContext = new MockContext(mockSharedPreferences);
    }

    @Test
    public void testDefaultRawAudioModeWithNullContext() {
        boolean isRaw = AudioPreferences.isRawAudioEnabled(null);
        assertTrue("Domyślny tryb toru audio przy null Context musi być true (RAW)", isRaw);
    }

    @Test
    public void testDefaultRawAudioModeWithEmptyPreferences() {
        boolean isRaw = AudioPreferences.isRawAudioEnabled(mockContext);
        assertTrue("Domyślny tryb toru audio w pustych preferencjach musi być true (RAW)", isRaw);
    }

    @Test
    public void testSetAndGetRawAudioPreferencesRoundtrip() {
        // Początkowo true
        assertTrue(AudioPreferences.isRawAudioEnabled(mockContext));

        // Wyłączenie trybu RAW (przełączenie na standardowy)
        AudioPreferences.setRawAudioEnabled(mockContext, false);
        assertFalse("Po wyłączeniu tryb RAW powinien wynosić false", AudioPreferences.isRawAudioEnabled(mockContext));

        // Ponowne włączenie trybu RAW
        AudioPreferences.setRawAudioEnabled(mockContext, true);
        assertTrue("Po ponownym włączeniu tryb RAW powinien wynosić true", AudioPreferences.isRawAudioEnabled(mockContext));
    }

    @Test
    public void testSetRawAudioPreferencesNullContextHandling() {
        // Zapis przy null kontekście nie rzuca wyjątku
        AudioPreferences.setRawAudioEnabled(null, false);
        assertTrue(AudioPreferences.isRawAudioEnabled(null));
    }

    @Test
    public void testAudioRecorderRawModeToggle() {
        AudioRecorder recorder = new AudioRecorder();
        // Domyślny stan
        assertTrue("AudioRecorder powinien domyślnie startować w trybie RAW", recorder.isRawAudioMode());

        // Przełączenie na false
        recorder.setRawAudioMode(false);
        assertFalse("AudioRecorder powinien zaakceptować zmianę na false", recorder.isRawAudioMode());

        // Przełączenie z powrotem na true
        recorder.setRawAudioMode(true);
        assertTrue("AudioRecorder powinien zaakceptować powrót na true", recorder.isRawAudioMode());
    }

    // =========================================================================
    // Mocki pomocnicze dla SharedPreferences (izolowane środowisko JVM)
    // =========================================================================

    private static class MockContext extends ContextWrapper {
        private final SharedPreferences preferences;

        MockContext(SharedPreferences preferences) {
            super(null);
            this.preferences = preferences;
        }

        @Override
        public SharedPreferences getSharedPreferences(String name, int mode) {
            return preferences;
        }

        @Override
        public Context getApplicationContext() {
            return this;
        }
    }

    private static class MockSharedPreferences implements SharedPreferences {
        private final Map<String, Object> storage = new HashMap<>();

        @Override
        public Map<String, ?> getAll() {
            return new HashMap<>(storage);
        }

        @Override
        public String getString(String key, String defValue) {
            Object val = storage.get(key);
            return val instanceof String ? (String) val : defValue;
        }

        @Override
        public Set<String> getStringSet(String key, Set<String> defValues) {
            return defValues;
        }

        @Override
        public int getInt(String key, int defValue) {
            Object val = storage.get(key);
            return val instanceof Integer ? (Integer) val : defValue;
        }

        @Override
        public long getLong(String key, long defValue) {
            Object val = storage.get(key);
            return val instanceof Long ? (Long) val : defValue;
        }

        @Override
        public float getFloat(String key, float defValue) {
            Object val = storage.get(key);
            return val instanceof Float ? (Float) val : defValue;
        }

        @Override
        public boolean getBoolean(String key, boolean defValue) {
            Object val = storage.get(key);
            return val instanceof Boolean ? (Boolean) val : defValue;
        }

        @Override
        public boolean contains(String key) {
            return storage.containsKey(key);
        }

        @Override
        public Editor edit() {
            return new MockEditor(storage);
        }

        @Override
        public void registerOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {}

        @Override
        public void unregisterOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {}
    }

    private static class MockEditor implements SharedPreferences.Editor {
        private final Map<String, Object> storage;
        private final Map<String, Object> pending = new HashMap<>();

        MockEditor(Map<String, Object> storage) {
            this.storage = storage;
        }

        @Override
        public SharedPreferences.Editor putString(String key, String value) {
            pending.put(key, value);
            return this;
        }

        @Override
        public SharedPreferences.Editor putStringSet(String key, Set<String> values) {
            pending.put(key, values);
            return this;
        }

        @Override
        public SharedPreferences.Editor putInt(String key, int value) {
            pending.put(key, value);
            return this;
        }

        @Override
        public SharedPreferences.Editor putLong(String key, long value) {
            pending.put(key, value);
            return this;
        }

        @Override
        public SharedPreferences.Editor putFloat(String key, float value) {
            pending.put(key, value);
            return this;
        }

        @Override
        public SharedPreferences.Editor putBoolean(String key, boolean value) {
            pending.put(key, value);
            return this;
        }

        @Override
        public SharedPreferences.Editor remove(String key) {
            pending.remove(key);
            storage.remove(key);
            return this;
        }

        @Override
        public SharedPreferences.Editor clear() {
            pending.clear();
            storage.clear();
            return this;
        }

        @Override
        public boolean commit() {
            apply();
            return true;
        }

        @Override
        public void apply() {
            storage.putAll(pending);
        }
    }
}
