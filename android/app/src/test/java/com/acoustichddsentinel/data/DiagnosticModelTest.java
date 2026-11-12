package com.acoustichddsentinel.data;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe modelu encji bazy SQLite DiagnosticModel.
 */
public class DiagnosticModelTest {

    @Test
    public void testDiagnosticModelFieldsAndFormatting() {
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

        // Test formatowania daty
        String formattedDate = model.getFormattedDate();
        assertNotNull(formattedDate);
        assertTrue("Sformatowana data powinna zawierać dwukropek godziny", formattedDate.contains(":"));
    }
}
