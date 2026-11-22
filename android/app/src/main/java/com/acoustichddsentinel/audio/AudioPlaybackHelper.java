package com.acoustichddsentinel.audio;

import android.media.AudioAttributes;
import android.media.MediaPlayer;
import android.util.Log;

import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;

/**
 * Bezpieczny pomocnik odtwarzania nagrań audio diagnostyki HDD (android.media.MediaPlayer).
 * Zapewnia:
 * 1. Prawidłowy cykl życia odtwarzacza (start, stop, pauza, zwolnienie zasobów w onDestroy/onPause).
 * 2. Unikanie wycieków AudioFlinger / pamięci natywnej Androida.
 * 3. Obsługę asynchronicznego zakończenia i błędów odtwarzacza.
 * 4. Płynne przełączanie między próbkami w liście historii.
 */
public class AudioPlaybackHelper {

    private static final String TAG = "AudioPlaybackHelper";

    public interface PlaybackEventListener {
        void onPlaybackStarted(String filePath);
        void onPlaybackStopped(String filePath);
        void onPlaybackCompleted(String filePath);
        void onPlaybackError(String filePath, String errorMessage);
    }

    private MediaPlayer mediaPlayer;
    private String currentPlayingPath = null;
    private PlaybackEventListener currentListener = null;

    public AudioPlaybackHelper() {
    }

    /**
     * Odtwarza plik WAV/audio wskazany przez filePath.
     * Jeśli inny plik był odtwarzany, zostaje on najpierw zatrzymany.
     * Jeśli ten sam plik jest w trakcie odtwarzania, zostaje zatrzymany (toggle).
     */
    public synchronized void togglePlayback(String filePath, PlaybackEventListener listener) {
        if (filePath == null || filePath.trim().isEmpty()) {
            if (listener != null) {
                listener.onPlaybackError(filePath, "Brak ścieżki pliku audio.");
            }
            return;
        }

        File file = new File(filePath);
        if (!file.exists() || file.length() == 0) {
            if (listener != null) {
                listener.onPlaybackError(filePath, "Plik nagrania nie istnieje lub jest pusty.");
            }
            return;
        }

        if (isPlaying() && filePath.equals(currentPlayingPath)) {
            // Ten sam plik - zatrzymujemy
            stop();
            return;
        }

        // Zatrzymujemy ewentualne poprzednie odtwarzanie
        stop();

        this.currentPlayingPath = filePath;
        this.currentListener = listener;

        try {
            mediaPlayer = new MediaPlayer();
            mediaPlayer.setAudioAttributes(
                new AudioAttributes.Builder()
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .build()
            );

            mediaPlayer.setDataSource(file.getAbsolutePath());

            mediaPlayer.setOnPreparedListener(mp -> {
                try {
                    mp.start();
                    if (currentListener != null) {
                        currentListener.onPlaybackStarted(currentPlayingPath);
                    }
                } catch (IllegalStateException e) {
                    Log.e(TAG, "Błąd startu odtwarzacza: " + e.getMessage(), e);
                    handleError("Nie można uruchomić odtwarzania.");
                }
            });

            mediaPlayer.setOnCompletionListener(mp -> {
                String finishedPath = currentPlayingPath;
                PlaybackEventListener cb = currentListener;
                cleanupPlayer();
                if (cb != null) {
                    cb.onPlaybackCompleted(finishedPath);
                }
            });

            mediaPlayer.setOnErrorListener((mp, what, extra) -> {
                Log.e(TAG, "MediaPlayer błąd: what=" + what + ", extra=" + extra);
                handleError("Błąd dekodera audio (" + what + ", " + extra + ")");
                return true;
            });

            mediaPlayer.prepareAsync();

        } catch (IOException | IllegalStateException e) {
            Log.e(TAG, "Błąd konfiguracji MediaPlayer dla: " + filePath, e);
            handleError("Błąd wejścia/wyjścia podczas wczytywania audio: " + e.getMessage());
        }
    }

    private synchronized void handleError(String errorMessage) {
        String failedPath = currentPlayingPath;
        PlaybackEventListener cb = currentListener;
        cleanupPlayer();
        if (cb != null) {
            cb.onPlaybackError(failedPath, errorMessage);
        }
    }

    /**
     * Zatrzymuje bieżące odtwarzanie i powiadamia listener.
     */
    public synchronized void stop() {
        if (mediaPlayer != null) {
            String stoppedPath = currentPlayingPath;
            PlaybackEventListener cb = currentListener;
            cleanupPlayer();
            if (cb != null && stoppedPath != null) {
                cb.onPlaybackStopped(stoppedPath);
            }
        }
    }

    /**
     * Zwraca true jeśli odtwarzacz jest aktualnie aktywny i odtwarza dźwięk.
     */
    public synchronized boolean isPlaying() {
        if (mediaPlayer == null) {
            return false;
        }
        try {
            return mediaPlayer.isPlaying();
        } catch (IllegalStateException e) {
            return false;
        }
    }

    /**
     * Sprawdza czy dany plik jest aktualnie odtwarzany.
     */
    public synchronized boolean isPlaying(String filePath) {
        return isPlaying() && filePath != null && filePath.equals(currentPlayingPath);
    }

    @Nullable
    public synchronized String getCurrentPlayingPath() {
        return currentPlayingPath;
    }

    private synchronized void cleanupPlayer() {
        if (mediaPlayer != null) {
            try {
                if (mediaPlayer.isPlaying()) {
                    mediaPlayer.stop();
                }
            } catch (Exception ignored) {
            }
            try {
                mediaPlayer.reset();
                mediaPlayer.release();
            } catch (Exception ignored) {
            }
            mediaPlayer = null;
        }
        currentPlayingPath = null;
        currentListener = null;
    }

    /**
     * Całkowite zwolnienie zasobów MediaPlayer. Należy wywołać w onDestroy / onStop aktywności.
     */
    public synchronized void release() {
        cleanupPlayer();
    }
}
