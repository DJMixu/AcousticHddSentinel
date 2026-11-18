package com.acoustichddsentinel.ml;

/**
 * Wspólny interfejs strategii wnioskowania dla modeli wykrywania awarii HDD w ONNX Runtime.
 * Zgodny z ADR-002 (Wzorzec Strategii, 100% czysta Java 17, Zero Kotlin).
 */
public interface DiskFailureDetector extends AutoCloseable {

    /**
     * Zwraca czytelną dla użytkownika nazwę modelu (np. "Model Klasyczny (CNN)").
     */
    String getModelName();

    /**
     * Zwraca szczegółowy opis techniczny modelu i architektury.
     */
    String getModelDescription();

    /**
     * Zwraca powiązany typ modelu (enum ModelType).
     */
    ModelType getModelType();

    /**
     * Sprawdza, czy sesja ONNX Runtime modelu jest załadowana i gotowa do inferencji.
     */
    boolean isLoaded();

    /**
     * Jawnie ładuje model do pamięci urządzenia (lazy loading).
     *
     * @throws Exception w przypadku błędu ładowania zasobów lub sesji ONNX Runtime
     */
    void load() throws Exception;

    /**
     * Wykonuje wnioskowanie dla pojedynczego 3-kanałowego tensora [1, 3, 128, 259].
     * Jeśli model nie jest jeszcze załadowany, metoda powinna go załadować (lazy-loading).
     *
     * @param flatTensor spłaszczony bufor float[] o długości 99456 (3 * 128 * 259)
     * @return wynik klasyfikacji z rozkładem prawdopodobieństw i metrykami
     * @throws Exception w przypadku błędu inferencji lub sesji ONNX Runtime
     */
    ClassificationResult classifyChunk(float[] flatTensor) throws Exception;

    /**
     * Zamyka sesję ONNX Runtime i natychmiast zwalnia pamięć RAM/NPU urządzenia.
     */
    @Override
    void close();
}
