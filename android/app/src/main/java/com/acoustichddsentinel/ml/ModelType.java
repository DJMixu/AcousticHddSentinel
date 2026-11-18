package com.acoustichddsentinel.ml;

/**
 * Typy modeli inferencyjnych wspieranych w architekturze Strategy Pattern.
 * Zgodność z ADR-002, 100% czysta Java 17.
 */
public enum ModelType {
    LEGACY("Model Klasyczny (CNN)", "Sprawdzony model 2D CNN (4 klasy: beeping, clicking, normal, scratching) z optymalizacją INT8."),
    ALTERNATIVE("Model Alternatywny", "Eksperymentalny model CNN z separacją prototypów OOD i zaawansowaną detekcją anomalii.");

    private final String title;
    private final String description;

    ModelType(String title, String description) {
        this.title = title;
        this.description = description;
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }
}
