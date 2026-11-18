package com.acoustichddsentinel.ml;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * Zarządzanie trwałym zapisem preferencji wyboru modelu w SharedPreferences.
 * Zgodnie z wytycznymi architektonicznymi, domyślnie ZAWSZE zwraca ModelType.LEGACY.
 */
public final class ModelPreferences {

    private static final String TAG = "ModelPreferences";
    private static final String PREF_NAME = "acoustic_hdd_model_preferences";
    private static final String KEY_SELECTED_MODEL = "selected_model_type";

    private ModelPreferences() {
        // Blokada instancjonowania klasy narzędziowej
    }

    /**
     * Pobiera aktualnie wybrany typ modelu. Domyślnie zawsze ModelType.LEGACY.
     *
     * @param context kontekst aplikacji Android
     * @return wybrany ModelType (nigdy null)
     */
    public static ModelType getSelectedModel(Context context) {
        if (context == null) {
            return ModelType.LEGACY;
        }
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            String savedName = prefs.getString(KEY_SELECTED_MODEL, ModelType.LEGACY.name());
            return ModelType.valueOf(savedName);
        } catch (Exception e) {
            Log.w(TAG, "Błąd odczytu preferencji modelu, powrót do domyślnego LEGACY: " + e.getMessage());
            return ModelType.LEGACY;
        }
    }

    /**
     * Zapisuje wybrany typ modelu w pamięci trwałej urządzenia.
     *
     * @param context   kontekst aplikacji Android
     * @param modelType typ modelu do zapisania (jeśli null, zapisuje LEGACY)
     */
    public static void setSelectedModel(Context context, ModelType modelType) {
        if (context == null) {
            return;
        }
        ModelType safeType = (modelType != null) ? modelType : ModelType.LEGACY;
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            prefs.edit().putString(KEY_SELECTED_MODEL, safeType.name()).apply();
            Log.i(TAG, "Zapisano preferencję wybranego modelu: " + safeType.name());
        } catch (Exception e) {
            Log.e(TAG, "Błąd zapisu preferencji modelu: " + e.getMessage(), e);
        }
    }
}
