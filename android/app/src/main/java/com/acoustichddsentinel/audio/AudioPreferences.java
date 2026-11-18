package com.acoustichddsentinel.audio;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * Pamięć trwała konfiguracji toru rejestracji audio w SharedPreferences.
 * Umożliwia włączenie opcjonalnego trybu czystego sygnału RAW (omijającego
 * systemowe filtry odszumiania, kasowania echa oraz automatycznej regulacji wzmocnienia).
 *
 * Zgodność z ADR-002 (Czysta Java 17, Zero Kotlin).
 */
public final class AudioPreferences {

    private static final String TAG = "AudioPreferences";
    private static final String PREF_NAME = "acoustic_hdd_audio_preferences";
    private static final String KEY_RAW_AUDIO_MODE = "key_raw_audio_mode";

    // Domyślnie opcja czystego toru RAW jest zalecana i aktywna dla optymalnej detekcji RPM i transjentów
    public static final boolean DEFAULT_RAW_AUDIO_MODE = true;

    private AudioPreferences() {
        // Blokada instancjonowania klasy narzędziowej
    }

    /**
     * Sprawdza, czy włączony jest opcjonalny tryb czystego toru audio RAW.
     *
     * @param context kontekst aplikacji Android
     * @return true jeśli włączony jest tryb RAW (Unprocessed Audio + wyłączone filtry systemowe)
     */
    public static boolean isRawAudioEnabled(Context context) {
        if (context == null) {
            return DEFAULT_RAW_AUDIO_MODE;
        }
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            return prefs.getBoolean(KEY_RAW_AUDIO_MODE, DEFAULT_RAW_AUDIO_MODE);
        } catch (Exception e) {
            Log.w(TAG, "Błąd odczytu preferencji toru audio, powrót do wartości domyślnej: " + e.getMessage());
            return DEFAULT_RAW_AUDIO_MODE;
        }
    }

    /**
     * Zapisuje wybór trybu audio w SharedPreferences.
     *
     * @param context kontekst aplikacji Android
     * @param enabled czy tryb RAW ma być aktywny
     */
    public static void setRawAudioEnabled(Context context, boolean enabled) {
        if (context == null) {
            return;
        }
        try {
            SharedPreferences prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            prefs.edit().putBoolean(KEY_RAW_AUDIO_MODE, enabled).apply();
            Log.i(TAG, "Zaktualizowano preferencję toru audio RAW: " + enabled);
        } catch (Exception e) {
            Log.e(TAG, "Błąd zapisu preferencji toru audio: " + e.getMessage(), e);
        }
    }
}
