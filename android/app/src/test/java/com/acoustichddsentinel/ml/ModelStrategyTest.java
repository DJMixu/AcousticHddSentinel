package com.acoustichddsentinel.ml;

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
 * Zestaw testów jednostkowych dla wzorca Strategii, zarządzania modelami AI
 * oraz preferencji w czystej Javie 17 (ADR-002, Zero Kotlin).
 *
 * Testuje:
 * 1. Domyślny wybór modelu (musi być zawsze LEGACY).
 * 2. Odczyt i zapis w ModelPreferences oraz odporność na błędy i null.
 * 3. Mechanizm lazy loading (modele nie ładują zasobów przy instancjonowaniu).
 * 4. Przełączanie między LegacyCnnDetector a AlternativeCnnDetector z uwalnianiem zasobów.
 * 5. Poprawność kontraktu interfejsu DiskFailureDetector i ClassificationResult.
 */
public class ModelStrategyTest {

    private MockSharedPreferences mockSharedPreferences;
    private MockContext mockContext;

    @Before
    public void setUp() {
        mockSharedPreferences = new MockSharedPreferences();
        mockContext = new MockContext(mockSharedPreferences);
    }

    // =========================================================================
    // 1. Domyślny wybór modelu (musi być zawsze LEGACY)
    // =========================================================================

    @Test
    public void testDefaultModelChoiceWithNullContext() {
        ModelType defaultType = ModelPreferences.getSelectedModel(null);
        assertEquals("Domyślny model przy braku kontekstu musi być LEGACY", ModelType.LEGACY, defaultType);
    }

    @Test
    public void testDefaultModelChoiceWithEmptyPreferences() {
        ModelType defaultType = ModelPreferences.getSelectedModel(mockContext);
        assertEquals("Domyślny model w pustych preferencjach musi być LEGACY", ModelType.LEGACY, defaultType);
    }

    @Test
    public void testModelStrategyManagerDefaultActiveModel() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);
        assertEquals("ModelStrategyManager musi domyślnie startować z modelem LEGACY",
            ModelType.LEGACY, manager.getActiveModelType());
        assertNotNull("Aktywny detektor nie może być null", manager.getActiveDetector());
        assertEquals("Aktywny detektor musi reprezentować model LEGACY",
            ModelType.LEGACY, manager.getActiveDetector().getModelType());
    }

    @Test
    public void testModelTypeEnumTitlesAndDescriptions() {
        assertEquals("Model Klasyczny (CNN)", ModelType.LEGACY.getTitle());
        assertTrue(ModelType.LEGACY.getDescription().contains("4 klasy"));

        assertEquals("Model Alternatywny", ModelType.ALTERNATIVE.getTitle());
        assertTrue(ModelType.ALTERNATIVE.getDescription().contains("model CNN"));
    }

    // =========================================================================
    // 2. Odczyt i zapis w ModelPreferences
    // =========================================================================

    @Test
    public void testModelPreferencesReadAndWrite() {
        // Początkowo LEGACY
        assertEquals(ModelType.LEGACY, ModelPreferences.getSelectedModel(mockContext));

        // Zapis ALTERNATIVE
        ModelPreferences.setSelectedModel(mockContext, ModelType.ALTERNATIVE);
        assertEquals(ModelType.ALTERNATIVE, ModelPreferences.getSelectedModel(mockContext));

        // Powrót do LEGACY
        ModelPreferences.setSelectedModel(mockContext, ModelType.LEGACY);
        assertEquals(ModelType.LEGACY, ModelPreferences.getSelectedModel(mockContext));
    }

    @Test
    public void testModelPreferencesNullHandling() {
        // Zapis przy null kontekście nie rzuca wyjątku
        ModelPreferences.setSelectedModel(null, ModelType.ALTERNATIVE);
        assertEquals(ModelType.LEGACY, ModelPreferences.getSelectedModel(null));

        // Zapis wartości null skutkuje bezpiecznym zapisem LEGACY
        ModelPreferences.setSelectedModel(mockContext, null);
        assertEquals(ModelType.LEGACY, ModelPreferences.getSelectedModel(mockContext));
    }

    @Test
    public void testModelPreferencesCorruptedValueRecovery() {
        // Zapis uszkodzonej wartości nieznanej w enumie
        mockSharedPreferences.edit().putString("selected_model_type", "UNKNOWN_FUTURE_MODEL").apply();

        // Powinien bezpiecznie powrócić do LEGACY
        ModelType recovered = ModelPreferences.getSelectedModel(mockContext);
        assertEquals("Przy błędnej nazwie modelu w preferencjach musi nastąpić fallback do LEGACY",
            ModelType.LEGACY, recovered);
    }

    // =========================================================================
    // 3. Mechanizm Lazy Loading
    // =========================================================================

    @Test
    public void testLazyLoadingInitialStateLegacy() {
        LegacyCnnDetector legacyDetector = new LegacyCnnDetector(mockContext);
        assertFalse("Nowo utworzony LegacyCnnDetector nie powinien być od razu załadowany (Lazy Loading)",
            legacyDetector.isLoaded());
    }

    @Test
    public void testLazyLoadingInitialStateAlternative() {
        AlternativeCnnDetector altDetector = new AlternativeCnnDetector(mockContext);
        assertFalse("Nowo utworzony AlternativeCnnDetector nie powinien być od razu załadowany (Lazy Loading)",
            altDetector.isLoaded());
    }

    @Test
    public void testLazyLoadingInModelStrategyManager() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);

        DiskFailureDetector legacy = manager.getDetector(ModelType.LEGACY);
        DiskFailureDetector alt = manager.getDetector(ModelType.ALTERNATIVE);

        assertNotNull(legacy);
        assertNotNull(alt);
        assertFalse("Model klasyczny w managerze nie może być ładowany przedwcześnie", legacy.isLoaded());
        assertFalse("Model alternatywny w managerze nie może być ładowany przedwcześnie", alt.isLoaded());
    }

    // =========================================================================
    // 4. Przełączanie między modelami i uwalnianie sesji
    // =========================================================================

    @Test
    public void testModelStrategyManagerSwitching() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);

        assertEquals(ModelType.LEGACY, manager.getActiveModelType());
        assertEquals("Model Klasyczny (CNN)", manager.getActiveDetector().getModelName());

        // Przełącz na ALTERNATIVE
        manager.setModelType(ModelType.ALTERNATIVE);
        assertEquals(ModelType.ALTERNATIVE, manager.getActiveModelType());
        assertEquals("Model Alternatywny", manager.getActiveDetector().getModelName());

        // Sprawdź czy preferencje zostały zaktualizowane
        assertEquals(ModelType.ALTERNATIVE, ModelPreferences.getSelectedModel(mockContext));

        // Przełącz z powrotem na LEGACY
        manager.setModelType(ModelType.LEGACY);
        assertEquals(ModelType.LEGACY, manager.getActiveModelType());
        assertEquals(ModelType.LEGACY, ModelPreferences.getSelectedModel(mockContext));
    }

    @Test
    public void testModelStrategyManagerIdempotentSwitch() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);

        DiskFailureDetector detectorBefore = manager.getActiveDetector();
        manager.setModelType(ModelType.LEGACY);
        DiskFailureDetector detectorAfter = manager.getActiveDetector();

        assertEquals("Przełączenie na ten sam model nie powinno zmieniać referencji detektora",
            detectorBefore, detectorAfter);
    }

    @Test
    public void testModelStrategyManagerNullTypeDefaultsToLegacy() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);
        manager.setModelType(ModelType.ALTERNATIVE);
        assertEquals(ModelType.ALTERNATIVE, manager.getActiveModelType());

        // Ustawienie null powinno ustawić LEGACY
        manager.setModelType(null);
        assertEquals(ModelType.LEGACY, manager.getActiveModelType());
    }

    @Test
    public void testModelStrategyManagerCloseLifecycle() {
        ModelStrategyManager manager = new ModelStrategyManager(mockContext);

        // Zamknięcie managera powinno wywołać close na obu detektorach bez wyjątków
        manager.close();

        assertFalse(manager.getDetector(ModelType.LEGACY).isLoaded());
        assertFalse(manager.getDetector(ModelType.ALTERNATIVE).isLoaded());

        // Podwójne zamknięcie powinno być bezpieczne i idempotentne
        manager.close();
    }

    // =========================================================================
    // 5. Poprawność kontraktu interfejsu DiskFailureDetector i ClassificationResult
    // =========================================================================

    @Test
    public void testDiskFailureDetectorContractLegacy() {
        DiskFailureDetector detector = new LegacyCnnDetector(mockContext);

        assertEquals("Model Klasyczny (CNN)", detector.getModelName());
        assertNotNull(detector.getModelDescription());
        assertFalse(detector.getModelDescription().isEmpty());
        assertEquals(ModelType.LEGACY, detector.getModelType());
        assertFalse(detector.isLoaded());

        // Metoda close na niezaładowanym detektorze nie może rzucać wyjątków
        detector.close();
        assertFalse(detector.isLoaded());
    }

    @Test
    public void testDiskFailureDetectorContractAlternative() {
        DiskFailureDetector detector = new AlternativeCnnDetector(mockContext);

        assertEquals("Model Alternatywny", detector.getModelName());
        assertNotNull(detector.getModelDescription());
        assertFalse(detector.getModelDescription().isEmpty());
        assertEquals(ModelType.ALTERNATIVE, detector.getModelType());
        assertFalse(detector.isLoaded());

        detector.close();
        assertFalse(detector.isLoaded());
    }

    @Test
    public void testCustomStrategyImplementationContract() throws Exception {
        // Weryfikacja rozszerzalności interfejsu DiskFailureDetector przez własną strategię testową
        TestMockDetector customDetector = new TestMockDetector(ModelType.LEGACY);

        assertFalse(customDetector.isLoaded());
        assertEquals(0, customDetector.loadCount);
        assertEquals(0, customDetector.classifyCount);

        customDetector.load();
        assertTrue(customDetector.isLoaded());
        assertEquals(1, customDetector.loadCount);

        float[] dummyTensor = new float[99456];
        ClassificationResult result = customDetector.classifyChunk(dummyTensor);

        assertNotNull(result);
        assertEquals("normal", result.getDominantClass());
        assertFalse(result.isAnomaly());
        assertEquals(1, customDetector.classifyCount);

        customDetector.close();
        assertFalse(customDetector.isLoaded());
        assertEquals(1, customDetector.closeCount);
    }

    @Test
    public void testClassificationResultContractAndFormatting() {
        Map<String, Float> probs = new HashMap<>();
        probs.put("normal", 0.965f);
        probs.put("clicking", 0.020f);
        probs.put("beeping", 0.010f);
        probs.put("scratching", 0.005f);

        ClassificationResult result = new ClassificationResult("normal", 0.965f, probs, false, 8);

        assertEquals("normal", result.getDominantClass());
        assertEquals(0.965f, result.getConfidence(), 1e-4f);
        assertEquals(probs, result.getProbabilities());
        assertFalse(result.isAnomaly());
        assertEquals(8, result.getInferenceTimeMs());
        assertEquals("96.5%", result.getFormattedConfidence());
        assertTrue(result.getRecommendation().contains("Dysk sprawny"));
    }

    @Test
    public void testClassificationResultAnomalies() {
        // Clicking
        ClassificationResult clicking = new ClassificationResult("clicking", 0.92f, null, true, 12);
        assertTrue(clicking.isAnomaly());
        assertTrue(clicking.getRecommendation().contains("Click of Death"));

        // Beeping
        ClassificationResult beeping = new ClassificationResult("beeping", 0.88f, null, true, 10);
        assertTrue(beeping.isAnomaly());
        assertTrue(beeping.getRecommendation().contains("Seized Spindle"));

        // Scratching
        ClassificationResult scratching = new ClassificationResult("scratching", 0.85f, null, true, 15);
        assertTrue(scratching.isAnomaly());
        assertTrue(scratching.getRecommendation().contains("Head Crash"));
    }

    // =========================================================================
    // Klasy pomocnicze Mock dla środowiska testowego JUnit
    // =========================================================================

    /**
     * Własna implementacja DiskFailureDetector do weryfikacji kontraktu wzorca Strategii.
     */
    private static class TestMockDetector implements DiskFailureDetector {
        private final ModelType modelType;
        private boolean loaded = false;
        int loadCount = 0;
        int classifyCount = 0;
        int closeCount = 0;

        TestMockDetector(ModelType modelType) {
            this.modelType = modelType;
        }

        @Override
        public String getModelName() {
            return "Test Mock Detector";
        }

        @Override
        public String getModelDescription() {
            return "Mock Detector dla testów jednostkowych kontraktu";
        }

        @Override
        public ModelType getModelType() {
            return modelType;
        }

        @Override
        public boolean isLoaded() {
            return loaded;
        }

        @Override
        public void load() {
            loadCount++;
            loaded = true;
        }

        @Override
        public ClassificationResult classifyChunk(float[] flatTensor) {
            if (!loaded) {
                load();
            }
            classifyCount++;
            Map<String, Float> map = new HashMap<>();
            map.put("normal", 0.99f);
            return new ClassificationResult("normal", 0.99f, map, false, 5);
        }

        @Override
        public void close() {
            closeCount++;
            loaded = false;
        }
    }

    /**
     * Mock ContextWrapper przekierowujący pobieranie SharedPreferences do pamięci RAM.
     */
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

    /**
     * W pełni w pamięciowa implementacja SharedPreferences dla testów jednostkowych.
     */
    private static class MockSharedPreferences implements SharedPreferences {
        final Map<String, Object> storage = new HashMap<>();

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
            return new MockEditor(this);
        }

        @Override
        public void registerOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {
        }

        @Override
        public void unregisterOnSharedPreferenceChangeListener(OnSharedPreferenceChangeListener listener) {
        }

        private static class MockEditor implements Editor {
            private final MockSharedPreferences parent;
            private final Map<String, Object> staging = new HashMap<>();

            MockEditor(MockSharedPreferences parent) {
                this.parent = parent;
            }

            @Override
            public Editor putString(String key, String value) {
                staging.put(key, value);
                return this;
            }

            @Override
            public Editor putStringSet(String key, Set<String> values) {
                staging.put(key, values);
                return this;
            }

            @Override
            public Editor putInt(String key, int value) {
                staging.put(key, value);
                return this;
            }

            @Override
            public Editor putLong(String key, long value) {
                staging.put(key, value);
                return this;
            }

            @Override
            public Editor putFloat(String key, float value) {
                staging.put(key, value);
                return this;
            }

            @Override
            public Editor putBoolean(String key, boolean value) {
                staging.put(key, value);
                return this;
            }

            @Override
            public Editor remove(String key) {
                staging.remove(key);
                parent.storage.remove(key);
                return this;
            }

            @Override
            public Editor clear() {
                staging.clear();
                parent.storage.clear();
                return this;
            }

            @Override
            public boolean commit() {
                parent.storage.putAll(staging);
                return true;
            }

            @Override
            public void apply() {
                commit();
            }
        }
    }
}
