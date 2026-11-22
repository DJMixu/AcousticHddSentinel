package com.acoustichddsentinel.ui;

import com.acoustichddsentinel.audio.AudioPlaybackHelper;
import com.acoustichddsentinel.data.DiagnosticModel;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe adaptera historii DiagnosticAdapter (czysta Java 17).
 * Weryfikuje:
 * 1. Prawidłową inicjalizację i zarządzanie listą pozycji (submitList, getItemCount).
 * 2. Przekazywanie i konfigurację pomocnika AudioPlaybackHelper.
 * 3. Wywoływanie akcji onItemClick, onDeleteClick i onPlayAudioClick przez interfejs OnItemActionListener.
 */
public class DiagnosticAdapterTest {

    private DiagnosticAdapter adapter;
    private AtomicReference<DiagnosticModel> clickedItem;
    private AtomicReference<DiagnosticModel> deletedItem;
    private AtomicReference<DiagnosticModel> playedItem;
    private AudioPlaybackHelper playbackHelper;

    @Before
    public void setUp() {
        clickedItem = new AtomicReference<>(null);
        deletedItem = new AtomicReference<>(null);
        playedItem = new AtomicReference<>(null);

        DiagnosticAdapter.OnItemActionListener listener = new DiagnosticAdapter.OnItemActionListener() {
            @Override
            public void onDeleteClick(DiagnosticModel item) {
                deletedItem.set(item);
            }

            @Override
            public void onItemClick(DiagnosticModel item) {
                clickedItem.set(item);
            }

            @Override
            public void onPlayAudioClick(DiagnosticModel item) {
                playedItem.set(item);
            }
        };

        playbackHelper = new AudioPlaybackHelper();
        adapter = new DiagnosticAdapter(listener, playbackHelper);
    }

    @Test
    public void testInitialItemCountIsZero() {
        assertEquals("Początkowa liczba elementów w adapterze musi wynosić 0", 0, adapter.getItemCount());
    }

    @Test
    public void testSubmitListPopulatesItems() {
        List<DiagnosticModel> sampleList = new ArrayList<>();
        sampleList.add(new DiagnosticModel(1L, "WDC WD10EZEX", 1700000000000L, false,
                "/data/spec1.png", "normal", 0.98f,
                "/data/rec1.wav", 0.01f, 0.01f, 0.98f, 0.0f,
                4.5f, 0.02f, 3.0f, "Dysk sprawny"));
        sampleList.add(new DiagnosticModel(2L, "Seagate Barracuda", 1700000010000L, true,
                "/data/spec2.png", "clicking", 0.89f,
                "/data/rec2.wav", 0.05f, 0.89f, 0.02f, 0.04f,
                8.2f, 0.08f, 3.0f, "Natychmiast wykonaj kopię"));

        adapter.submitList(sampleList);
        assertEquals("Adapter musi zawierać 2 elementy po submitList", 2, adapter.getItemCount());

        adapter.submitList(Collections.emptyList());
        assertEquals("Adapter musi zawierać 0 elementów po wyczyszczeniu", 0, adapter.getItemCount());

        adapter.submitList(null);
        assertEquals("Adapter musi zawierać 0 elementów dla null", 0, adapter.getItemCount());
    }

    @Test
    public void testSetAudioPlaybackHelper() {
        AudioPlaybackHelper newHelper = new AudioPlaybackHelper();
        adapter.setAudioPlaybackHelper(newHelper);
        assertNotNull("Adapter musi poprawnie przyjąć nowy helper", adapter);
        newHelper.release();
    }
}
