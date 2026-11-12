package com.acoustichddsentinel.ml;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Testy jednostkowe modelu wyników klasyfikacji ClassificationResult.
 */
public class ClassificationResultTest {

    @Test
    public void testNormalClassificationResult() {
        Map<String, Float> probs = new HashMap<>();
        probs.put("normal", 0.95f);
        probs.put("clicking", 0.02f);
        probs.put("beeping", 0.02f);
        probs.put("scratching", 0.01f);

        ClassificationResult result = new ClassificationResult("normal", 0.95f, probs, false, 12);

        assertEquals("normal", result.getDominantClass());
        assertEquals(0.95f, result.getConfidence(), 1e-4f);
        assertFalse("Dysk normalny nie powinien być oznaczony jako anomalia", result.isAnomaly());
        assertEquals(12, result.getInferenceTimeMs());
        assertEquals("95.0%", result.getFormattedConfidence());
        assertTrue(result.getRecommendation().contains("Dysk sprawny"));
    }

    @Test
    public void testClickingAnomalyResult() {
        ClassificationResult result = new ClassificationResult("clicking", 0.88f, null, true, 14);

        assertTrue("Clicking musi być oznaczony jako anomalia", result.isAnomaly());
        assertTrue("Rekomendacja musi zawierać ostrzeżenie o pozycjonerze / Click of Death",
            result.getRecommendation().contains("Click of Death"));
    }

    @Test
    public void testBeepingAnomalyResult() {
        ClassificationResult result = new ClassificationResult("beeping", 0.91f, null, true, 11);

        assertTrue(result.isAnomaly());
        assertTrue("Rekomendacja musi ostrzegać przed zatarciem silnika wrzeciona",
            result.getRecommendation().contains("Seized Spindle"));
    }

    @Test
    public void testScratchingAnomalyResult() {
        ClassificationResult result = new ClassificationResult("scratching", 0.79f, null, true, 15);

        assertTrue(result.isAnomaly());
        assertTrue("Rekomendacja musi ostrzegać przed tarciem głowic (Head Crash)",
            result.getRecommendation().contains("Head Crash"));
    }
}
