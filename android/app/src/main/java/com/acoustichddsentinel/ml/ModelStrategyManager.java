package com.acoustichddsentinel.ml;

import android.content.Context;
import android.util.Log;

/**
 * Menedżer wzorca Strategii dla modeli detekcji uszkodzeń HDD.
 * Odpowiada za:
 * 1. Zarządzanie cyklem życia modeli (Lazy Loading).
 * 2. Zwalnianie pamięci RAM/NPU (zamykanie sesji poprzedniego modelu przed aktywacją nowego).
 * 3. Trwały zapis aktywnego modelu w SharedPreferences (ModelPreferences).
 * 4. Transparentną delegację zapytań inferencyjnych.
 *
 * Zgodność z ADR-002, 100% Czysta Java 17.
 */
public class ModelStrategyManager implements AutoCloseable {

    private static final String TAG = "ModelStrategyManager";

    private final Context context;
    private final LegacyCnnDetector legacyDetector;
    private final AlternativeCnnDetector alternativeDetector;

    private ModelType activeModelType;
    private DiskFailureDetector activeDetector;

    public ModelStrategyManager(Context context) {
        this.context = context != null ? context.getApplicationContext() : null;
        this.legacyDetector = new LegacyCnnDetector(this.context);
        this.alternativeDetector = new AlternativeCnnDetector(this.context);

        this.activeModelType = ModelPreferences.getSelectedModel(this.context);
        this.activeDetector = getDetector(this.activeModelType);
        Log.i(TAG, "Zainicjalizowano ModelStrategyManager. Aktywny model z preferencji: " + activeModelType);
    }

    /**
     * Zwraca instancję detektora dla określonego typu modelu.
     *
     * @param type typ modelu
     * @return instancja DiskFailureDetector
     */
    public DiskFailureDetector getDetector(ModelType type) {
        if (type == ModelType.ALTERNATIVE) {
            return alternativeDetector;
        }
        return legacyDetector;
    }

    /**
     * Zwraca aktualnie aktywny typ modelu.
     */
    public synchronized ModelType getActiveModelType() {
        return activeModelType;
    }

    /**
     * Zwraca aktualnie aktywną strategię detektora.
     */
    public synchronized DiskFailureDetector getActiveDetector() {
        if (activeDetector == null) {
            activeDetector = getDetector(activeModelType);
        }
        return activeDetector;
    }

    /**
     * Przełącza aktywny model inferencyjny.
     * Zgodnie z ADR-002, natychmiast zamyka sesję ONNX Runtime poprzedniego modelu,
     * uwalniając pamięć RAM urządzenia przed aktywacją nowego modelu.
     *
     * @param newType nowy typ modelu (jeśli null, przyjmuje LEGACY)
     */
    public synchronized void setModelType(ModelType newType) {
        if (newType == null) {
            newType = ModelType.LEGACY;
        }

        if (this.activeModelType == newType && this.activeDetector != null) {
            Log.d(TAG, "Model " + newType + " jest już aktywny.");
            return;
        }

        // Zwalnianie pamięci poprzedniego aktywnego modelu
        if (this.activeDetector != null && this.activeDetector.isLoaded()) {
            Log.i(TAG, "Zwalnianie pamięci i zamykanie sesji poprzedniego modelu: " + this.activeDetector.getModelName());
            this.activeDetector.close();
        }

        this.activeModelType = newType;
        this.activeDetector = getDetector(newType);

        // Zapis do preferencji
        ModelPreferences.setSelectedModel(context, newType);
        Log.i(TAG, "Przełączono model na: " + newType.getTitle() + " (" + newType.name() + ")");
    }

    /**
     * Jawne załadowanie aktualnie aktywnego modelu w tle (np. przy starcie aplikacji).
     */
    public synchronized void loadActiveModel() throws Exception {
        DiskFailureDetector detector = getActiveDetector();
        if (!detector.isLoaded()) {
            detector.load();
        }
    }

    /**
     * Wykonuje wnioskowanie za pomocą aktualnie aktywnej strategii modelu.
     *
     * @param flatTensor spłaszczony bufor float[] o długości 99456
     * @return wynik klasyfikacji
     * @throws Exception w przypadku błędu inferencji lub sesji ONNX Runtime
     */
    public synchronized ClassificationResult classifyChunk(float[] flatTensor) throws Exception {
        DiskFailureDetector detector = getActiveDetector();
        return detector.classifyChunk(flatTensor);
    }

    @Override
    public synchronized void close() {
        Log.i(TAG, "Zamykanie wszystkich strategii modeli w ModelStrategyManager...");
        if (legacyDetector != null) {
            legacyDetector.close();
        }
        if (alternativeDetector != null) {
            alternativeDetector.close();
        }
    }
}
