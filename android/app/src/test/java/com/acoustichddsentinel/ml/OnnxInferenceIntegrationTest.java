package com.acoustichddsentinel.ml;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.acoustichddsentinel.dsp.AudioPreprocessor;
import com.acoustichddsentinel.dsp.DspConstants;
import com.acoustichddsentinel.dsp.DspPhysicalGuard;
import com.acoustichddsentinel.dsp.RpmHarmonicsAnalyzer;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Test weryfikacyjny inferencji ONNX Runtime bezpośrednio w środowisku testowym Androida (Java 17).
 * Sprawdza integrację potoku DSP -> ONNX INT8 -> DSP Physical Guard.
 */
public class OnnxInferenceIntegrationTest {

    private OrtEnvironment environment;
    private OrtSession sessionInt8;
    private OrtSession sessionFp32;
    private AudioPreprocessor preprocessor;
    private RpmHarmonicsAnalyzer rpmAnalyzer;
    private DspPhysicalGuard physicalGuard;

    @Before
    public void setUp() throws Exception {
        environment = OrtEnvironment.getEnvironment();

        File int8File = new File("src/main/assets/best_hdd_cnn_model_int8.onnx");
        if (!int8File.exists()) {
            int8File = new File("android/app/src/main/assets/best_hdd_cnn_model_int8.onnx");
        }
        assertTrue("Plik modelu INT8 musi istnieć: " + int8File.getAbsolutePath(), int8File.exists());

        File fp32File = new File("src/main/assets/best_hdd_cnn_model.onnx");
        if (!fp32File.exists()) {
            fp32File = new File("android/app/src/main/assets/best_hdd_cnn_model.onnx");
        }
        assertTrue("Plik modelu FP32 musi istnieć: " + fp32File.getAbsolutePath(), fp32File.exists());

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        
        // Zgodnie z architekturą AcousticClassifier i LegacyCnnDetector: próba INT8 z automatycznym fallbackiem na FP32
        try {
            sessionInt8 = environment.createSession(int8File.getAbsolutePath(), options);
        } catch (Exception e) {
            System.out.println("INT8 wymaga NNAPI/Android lub nie posiada ConvInteger na tym CPU (" + e.getMessage() + "), używam FP32 do testów inferencji.");
            sessionInt8 = null;
        }
        sessionFp32 = environment.createSession(fp32File.getAbsolutePath(), options);

        preprocessor = new AudioPreprocessor();
        rpmAnalyzer = new RpmHarmonicsAnalyzer();
        physicalGuard = new DspPhysicalGuard();
    }

    @After
    public void tearDown() throws Exception {
        if (sessionInt8 != null) sessionInt8.close();
        if (sessionFp32 != null) sessionFp32.close();
    }

    @Test
    public void testOnnxModelsLoadedSuccessfully() {
        assertNotNull("Sesja FP32 musi być zawsze aktywna", sessionFp32);
        assertEquals("Liczba wejść modelu musi wynosić 1", 1, sessionFp32.getNumInputs());
        assertEquals("Liczba wyjść modelu musi wynosić 1", 1, sessionFp32.getNumOutputs());
    }

    @Test
    public void testFullPipelineSyntheticNormalDrive() throws Exception {
        OrtSession activeSession = (sessionInt8 != null) ? sessionInt8 : sessionFp32;
        assertNotNull("Musi istnieć aktywna sesja inferencyjna", activeSession);

        // Syntetyczny sygnał zdrowego dysku 7200 RPM (f0 = 120.0 Hz)
        float[] samples = new float[DspConstants.CHUNK_SAMPLES];
        for (int i = 0; i < samples.length; i++) {
            double t = (double) i / DspConstants.SAMPLE_RATE;
            samples[i] = (float) (
                    0.25 * Math.sin(2.0 * Math.PI * 120.0 * t) +
                    0.15 * Math.sin(2.0 * Math.PI * 240.0 * t) +
                    0.10 * Math.sin(2.0 * Math.PI * 360.0 * t) +
                    0.05 * Math.sin(2.0 * Math.PI * 480.0 * t)
            );
        }

        // 1. DSP Preprocessing do tensora [3, 128, 259]
        float[] flatTensor = preprocessor.processAudioChunkToFlatTensor(samples);
        assertEquals(DspConstants.TENSOR_FLAT_SIZE, flatTensor.length);

        // 2. Wnioskowanie ONNX
        FloatBuffer buffer = FloatBuffer.wrap(flatTensor);
        long[] shape = new long[]{1, 3, 128, 259};
        float[] rawProbs;
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, buffer, shape);
             OrtSession.Result result = activeSession.run(Collections.singletonMap("input_audio_tensor", inputTensor))) {
            float[][] out = (float[][]) result.get(0).getValue();
            rawProbs = out[0];
        }

        assertEquals("Model musi zwrócić 4 prawdopodobieństwa", 4, rawProbs.length);
        float sum = rawProbs[0] + rawProbs[1] + rawProbs[2] + rawProbs[3];
        assertEquals("Suma prawdopodobieństw Softmax ~1.0", 1.0f, sum, 0.05f);

        // 3. DSP Strażnik i RPM
        RpmHarmonicsAnalyzer.RpmResult rpmRes = rpmAnalyzer.analyzeChunk(samples);
        assertTrue("Musi wykryć rotację 7200 RPM", rpmRes.hasRotation);
        assertEquals(7200, rpmRes.nominalRpm);

        float[][] logMelRaw = preprocessor.extractLogMelForVisualization(samples);
        DspPhysicalGuard.SanitizationResult guardRes = physicalGuard.sanitizeProbabilities(
                rawProbs, samples, logMelRaw, rpmRes
        );

        // Po DSP: z potwierdzoną rotacją 7200 RPM, prawdopodobieństwo 'normal' musi wynosić >= 0.85
        assertTrue("DSP musi potwierdzić stan normalny z rotacją", guardRes.sanitizedProbs[2] >= 0.85f);
        float faultProb = guardRes.sanitizedProbs[0] + guardRes.sanitizedProbs[1] + guardRes.sanitizedProbs[3];
        assertTrue("DSP musi stłumić prawdopodobieństwo awarii poniżej 0.15", faultProb < 0.15f);
    }
}
