package com.acoustichddsentinel;

import android.app.Application;
import android.util.Log;

/**
 * Główna klasa aplikacji Android (Application Singleton).
 */
public class AcousticSentinelApp extends Application {

    private static final String TAG = "AcousticSentinelApp";

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG, "Acoustic HDD Sentinel zainicjalizowany pomyślnie. Tryb 100% OFFLINE aktywny.");
    }
}
