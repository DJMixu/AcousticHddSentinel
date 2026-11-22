package com.acoustichddsentinel.data;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe modelu encji bazy SQLite DiagnosticModel (Wersja v1 i wzbogacona v2).
 * Weryfikuje wszystkie gettery, settery, konstruktory v1/v2 oraz formatowanie daty.
 */
public class DiagnosticModelTest {

    @Test
    public void testDiagnosticModelLegacyFieldsAndFormatting() {
        long timestamp = 1731376800000L; // Stały timestamp
        DiagnosticModel model = new DiagnosticModel(
            42,
            "Seagate Barracuda 2TB",
            timestamp,
            true,
            "/data/user/0/com.acoustichddsentinel/files/spectrograms/spec_clicking_1.png",
            "clicking",
            0.942f
        );

        assertEquals(42, model.getId());
        assertEquals("Seagate Barracuda 2TB", model.getHddName());
        assertEquals(timestamp, model.getTimestamp());
        assertTrue("hasAnomaly musi być true", model.hasAnomaly());
        assertEquals("clicking", model.getPredictedClass());
        assertEquals(0.942f, model.getConfidence(), 1e-4f);
        assertNotNull(model.getSpectrogramPath());

        // Domyślne wartości nowych pól w konstruktorze legacy
        assertNull(model.getAudioPath());
        assertEquals(0.0f, model.getProbBeeping(), 1e-6f);
        assertEquals(0.0f, model.getProbClicking(), 1e-6f);
        assertEquals(0.0f, model.getProbNormal(), 1e-6f);
        assertEquals(0.0f, model.getProbScratching(), 1e-6f);
        assertEquals(0.0f, model.getCrestFactor(), 1e-6f);
        assertEquals(0.0f, model.getPeakRms(), 1e-6f);
        assertEquals(3.0f, model.getScanDurationS(), 1e-6f);
        assertNull(model.getRecommendation());

        // Test formatowania daty
        String formattedDate = model.getFormattedDate();
        assertNotNull(formattedDate);
        assertTrue("Sformatowana data powinna zawierać dwukropek godziny", formattedDate.contains(":"));
    }

    @Test
    public void testDiagnosticModelEnrichedV2Fields() {
        long timestamp = 1731376800000L;
        DiagnosticModel model = new DiagnosticModel(
            101,
            "WD Red Plus 4TB",
            timestamp,
            true,
            "/path/to/spec.png",
            "clicking",
            0.88f,
            "/path/to/rec_clicking.wav",
            0.05f, // probBeeping
            0.88f, // probClicking
            0.03f, // probNormal
            0.04f, // probScratching
            6.85f, // crestFactor
            0.145f, // peakRms
            4.5f,  // scanDurationS
            "Wykryto powtarzalne uderzenia pozycjonera głowic."
        );

        assertEquals(101, model.getId());
        assertEquals("WD Red Plus 4TB", model.getHddName());
        assertEquals(timestamp, model.getTimestamp());
        assertTrue(model.hasAnomaly());
        assertEquals("/path/to/spec.png", model.getSpectrogramPath());
        assertEquals("clicking", model.getPredictedClass());
        assertEquals(0.88f, model.getConfidence(), 1e-4f);

        // Weryfikacja nowych pól schematu v2
        assertEquals("/path/to/rec_clicking.wav", model.getAudioPath());
        assertEquals(0.05f, model.getProbBeeping(), 1e-4f);
        assertEquals(0.88f, model.getProbClicking(), 1e-4f);
        assertEquals(0.03f, model.getProbNormal(), 1e-4f);
        assertEquals(0.04f, model.getProbScratching(), 1e-4f);
        assertEquals(6.85f, model.getCrestFactor(), 1e-4f);
        assertEquals(0.145f, model.getPeakRms(), 1e-4f);
        assertEquals(4.5f, model.getScanDurationS(), 1e-4f);
        assertEquals("Wykryto powtarzalne uderzenia pozycjonera głowic.", model.getRecommendation());

        // Test wszystkich setterów
        model.setId(202);
        assertEquals(202, model.getId());

        model.setAudioPath("/new/path.wav");
        assertEquals("/new/path.wav", model.getAudioPath());

        model.setProbBeeping(0.12f);
        assertEquals(0.12f, model.getProbBeeping(), 1e-4f);

        model.setProbClicking(0.75f);
        assertEquals(0.75f, model.getProbClicking(), 1e-4f);

        model.setProbNormal(0.08f);
        assertEquals(0.08f, model.getProbNormal(), 1e-4f);

        model.setProbScratching(0.05f);
        assertEquals(0.05f, model.getProbScratching(), 1e-4f);

        model.setCrestFactor(8.12f);
        assertEquals(8.12f, model.getCrestFactor(), 1e-4f);

        model.setPeakRms(0.200f);
        assertEquals(0.200f, model.getPeakRms(), 1e-4f);

        model.setScanDurationS(6.0f);
        assertEquals(6.0f, model.getScanDurationS(), 1e-4f);

        model.setRecommendation("Nowa rekomendacja");
        assertEquals("Nowa rekomendacja", model.getRecommendation());
    }

    @Test
    public void testDiagnosticModelConstructorsWithoutId() {
        long timestamp = 1731390000000L;

        // Konstruktor v1 bez ID (domyślnie id = 0)
        DiagnosticModel v1Model = new DiagnosticModel(
            "Toshiba DT01ACA",
            timestamp,
            false,
            "/path/spec_normal.png",
            "normal",
            0.99f
        );
        assertEquals(0, v1Model.getId());
        assertEquals("Toshiba DT01ACA", v1Model.getHddName());
        assertEquals(timestamp, v1Model.getTimestamp());
        assertEquals("normal", v1Model.getPredictedClass());
        assertNull(v1Model.getAudioPath());

        // Konstruktor v2 bez ID (domyślnie id = 0)
        DiagnosticModel v2Model = new DiagnosticModel(
            "HGST Ultrastar 4TB",
            timestamp,
            true,
            "/path/spec_scratch.png",
            "scratching",
            0.91f,
            "/path/rec_scratch.wav",
            0.01f, // probBeeping
            0.03f, // probClicking
            0.05f, // probNormal
            0.91f, // probScratching
            3.45f, // crestFactor
            0.180f, // peakRms
            3.0f, // scanDurationS
            "Ryzyko degradacji powierzchni talerza magnetycznego."
        );
        assertEquals(0, v2Model.getId());
        assertEquals("HGST Ultrastar 4TB", v2Model.getHddName());
        assertEquals("/path/rec_scratch.wav", v2Model.getAudioPath());
        assertEquals(0.91f, v2Model.getProbScratching(), 1e-4f);
        assertEquals(3.45f, v2Model.getCrestFactor(), 1e-4f);
        assertEquals("Ryzyko degradacji powierzchni talerza magnetycznego.", v2Model.getRecommendation());
    }
}
