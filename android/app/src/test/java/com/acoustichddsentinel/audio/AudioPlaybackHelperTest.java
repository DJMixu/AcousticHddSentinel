package com.acoustichddsentinel.audio;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe AudioPlaybackHelper weryfikujące:
 * 1. Prawidłowy stan początkowy (brak odtwarzania, null path).
 * 2. Zabezpieczenie przed brakującym plikiem i null / empty path.
 * 3. Zabezpieczenie przed pustym plikiem (0 bajtów).
 * 4. Prawidłową obsługę metod stop() i release() bez wycieków pamięci.
 */
public class AudioPlaybackHelperTest {

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private AudioPlaybackHelper playbackHelper;

    @Before
    public void setUp() {
        playbackHelper = new AudioPlaybackHelper();
    }

    @After
    public void tearDown() {
        if (playbackHelper != null) {
            playbackHelper.release();
        }
    }

    @Test
    public void testInitialStateIsIdle() {
        assertFalse("Nowy helper nie może odtwarzać dźwięku", playbackHelper.isPlaying());
        assertNull("Ścieżka początkowa musi być null", playbackHelper.getCurrentPlayingPath());
        assertFalse("Sprawdzenie dowolnej ścieżki w stanie idle musi dać false",
                playbackHelper.isPlaying("/tmp/test.wav"));
    }

    @Test
    public void testNullOrEmptyPathReportsError() {
        AtomicBoolean errorCalled = new AtomicBoolean(false);
        AtomicReference<String> errorMsg = new AtomicReference<>();

        AudioPlaybackHelper.PlaybackEventListener listener = new AudioPlaybackHelper.PlaybackEventListener() {
            @Override
            public void onPlaybackStarted(String filePath) {}

            @Override
            public void onPlaybackStopped(String filePath) {}

            @Override
            public void onPlaybackCompleted(String filePath) {}

            @Override
            public void onPlaybackError(String filePath, String errorMessage) {
                errorCalled.set(true);
                errorMsg.set(errorMessage);
            }
        };

        playbackHelper.togglePlayback(null, listener);
        assertTrue("Null path musi zgłosić błąd przez listener", errorCalled.get());
        assertNotNull("Komunikat błędu nie może być null", errorMsg.get());

        errorCalled.set(false);
        playbackHelper.togglePlayback("   ", listener);
        assertTrue("Pusta ścieżka musi zgłosić błąd przez listener", errorCalled.get());
    }

    @Test
    public void testNonExistentFileReportsError() {
        AtomicBoolean errorCalled = new AtomicBoolean(false);
        AtomicReference<String> errorMsg = new AtomicReference<>();

        AudioPlaybackHelper.PlaybackEventListener listener = new AudioPlaybackHelper.PlaybackEventListener() {
            @Override
            public void onPlaybackStarted(String filePath) {}

            @Override
            public void onPlaybackStopped(String filePath) {}

            @Override
            public void onPlaybackCompleted(String filePath) {}

            @Override
            public void onPlaybackError(String filePath, String errorMessage) {
                errorCalled.set(true);
                errorMsg.set(errorMessage);
            }
        };

        String nonExistent = new File(tempFolder.getRoot(), "missing_rec.wav").getAbsolutePath();
        playbackHelper.togglePlayback(nonExistent, listener);

        assertTrue("Nieistniejący plik musi wywołać onPlaybackError", errorCalled.get());
        assertTrue("Komunikat błędu musi informować o braku pliku",
                errorMsg.get().contains("nie istnieje"));
    }

    @Test
    public void testZeroByteFileReportsError() throws IOException {
        File emptyFile = tempFolder.newFile("empty_recording.wav");
        assertEquals("Rozmiar pliku testowego musi wynosić 0", 0, emptyFile.length());

        AtomicBoolean errorCalled = new AtomicBoolean(false);
        AudioPlaybackHelper.PlaybackEventListener listener = new AudioPlaybackHelper.PlaybackEventListener() {
            @Override
            public void onPlaybackStarted(String filePath) {}

            @Override
            public void onPlaybackStopped(String filePath) {}

            @Override
            public void onPlaybackCompleted(String filePath) {}

            @Override
            public void onPlaybackError(String filePath, String errorMessage) {
                errorCalled.set(true);
            }
        };

        playbackHelper.togglePlayback(emptyFile.getAbsolutePath(), listener);
        assertTrue("Plik o zerowej długości musi wywołać błąd", errorCalled.get());
        assertFalse("Helper nie może odtwarzać pustego pliku", playbackHelper.isPlaying());
    }

    @Test
    public void testStopAndReleaseWhenIdleDoNotThrow() {
        // Wielokrotne wywołania stop i release nie mogą rzucać wyjątków
        playbackHelper.stop();
        playbackHelper.stop();
        playbackHelper.release();
        playbackHelper.release();

        assertFalse(playbackHelper.isPlaying());
        assertNull(playbackHelper.getCurrentPlayingPath());
    }
}
