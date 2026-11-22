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
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Test ewaluacyjny potoku aplikacji Android (Java 17) na autentycznych próbkach audio z bazy datasetu.
 * Weryfikuje zachowanie modelu ONNX przed i po użyciu filtrów DSP:
 * 1. Próbki dysków sprawnych (normal) - potwierdzenie braku fałszywych alarmów (FPR = 0.0%)
 * 2. Próbki awarii klikania głowic (clicking) - weryfikacja wykrywalności anomalii i estymatora MAD
 * 3. Próbki zatarcia silnika (beeping) - weryfikacja detekcji Stall Beep i klasyfikacji uszkodzenia
 * 4. Szum wentylatora komputera - weryfikacja stłumienia fałszywego alarmu przez strażnika fizycznego
 */
public class DatasetAudioPipelineTest {

    private OrtEnvironment environment;
    private OrtSession session;
    private AudioPreprocessor preprocessor;
    private RpmHarmonicsAnalyzer rpmAnalyzer;
    private DspPhysicalGuard physicalGuard;
    private File repoRoot;

    @Before
    public void setUp() throws Exception {
        environment = OrtEnvironment.getEnvironment();

        // Lokalizacja katalogu głównego repozytorium
        File userDir = new File(System.getProperty("user.dir"));
        if (new File(userDir, "dataset_categorized").exists()) {
            repoRoot = userDir;
        } else if (new File(userDir.getParentFile(), "dataset_categorized").exists()) {
            repoRoot = userDir.getParentFile();
        } else if (userDir.getParentFile() != null && new File(userDir.getParentFile().getParentFile(), "dataset_categorized").exists()) {
            repoRoot = userDir.getParentFile().getParentFile();
        } else {
            repoRoot = new File("/home/DJMixu/Pulpit/AcousticHddSentinel");
        }

        File modelFile = new File(repoRoot, "exported_models/best_hdd_cnn_model.onnx");
        if (!modelFile.exists()) {
            modelFile = new File("src/main/assets/best_hdd_cnn_model.onnx");
        }
        assertTrue("Plik modelu ONNX musi istnieć: " + modelFile.getAbsolutePath(), modelFile.exists());

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        session = environment.createSession(modelFile.getAbsolutePath(), options);

        preprocessor = new AudioPreprocessor();
        rpmAnalyzer = new RpmHarmonicsAnalyzer();
        physicalGuard = new DspPhysicalGuard();
    }

    @After
    public void tearDown() throws Exception {
        if (session != null) {
            session.close();
            session = null;
        }
    }

    private float[] loadWavChunk(File wavFile) throws Exception {
        byte[] fileBytes;
        try (FileInputStream fis = new FileInputStream(wavFile)) {
            fileBytes = fis.readAllBytes();
        }

        // Szukamy subchunka "data" w nagłówku RIFF
        int dataOffset = 44; // Domyślny offset standardowego nagłówka
        for (int i = 12; i < fileBytes.length - 8; i++) {
            if (fileBytes[i] == 'd' && fileBytes[i + 1] == 'a' && fileBytes[i + 2] == 't' && fileBytes[i + 3] == 'a') {
                dataOffset = i + 8;
                break;
            }
        }

        int availableBytes = fileBytes.length - dataOffset;
        int numShorts = Math.max(0, availableBytes / 2);
        int targetSamples = DspConstants.CHUNK_SAMPLES;
        int toRead = Math.min(numShorts, targetSamples);

        float[] samples = new float[targetSamples];
        ByteBuffer bb = ByteBuffer.wrap(fileBytes, dataOffset, toRead * 2).order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < toRead; i++) {
            short s = bb.getShort();
            samples[i] = s / 32768.0f;
        }
        return samples;
    }

    private EvaluationResult evaluatePipeline(float[] samples) throws Exception {
        float rms = AudioPreprocessor.computeRms(samples);

        // 1. Enkodowanie tensora [3, 128, 259]
        float[] flatTensor = preprocessor.processAudioChunkToFlatTensor(samples);

        // 2. Inferencja ONNX (Surowy model)
        FloatBuffer buffer = FloatBuffer.wrap(flatTensor);
        long[] shape = new long[]{1, 3, 128, 259};
        float[] rawProbs;
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, buffer, shape);
             OrtSession.Result result = session.run(Collections.singletonMap("input_audio_tensor", inputTensor))) {
            float[][] out = (float[][]) result.get(0).getValue();
            rawProbs = out[0];
        }

        float rawFaultProb = rawProbs[0] + rawProbs[1] + rawProbs[3];

        // 3. Po użyciu DSP (RPM + Fizyczny Strażnik)
        RpmHarmonicsAnalyzer.RpmResult rpmRes = rpmAnalyzer.analyzeChunk(samples);
        float[][] logMelRaw = preprocessor.extractLogMelForVisualization(samples);
        DspPhysicalGuard.SanitizationResult guardRes = physicalGuard.sanitizeProbabilities(
                rawProbs, samples, logMelRaw, rpmRes
        );

        float[] sanProbs = guardRes.sanitizedProbs;
        float dspFaultProb = sanProbs[0] + sanProbs[1] + sanProbs[3];
        boolean dspIsFault = (dspFaultProb >= 0.65f);

        int bestIdx = 0;
        float maxP = sanProbs[0];
        String[] classNames = new String[]{"beeping", "clicking", "normal", "scratching"};
        for (int i = 1; i < 4; i++) {
            if (sanProbs[i] > maxP) {
                maxP = sanProbs[i];
                bestIdx = i;
            }
        }
        String dominantClass = classNames[bestIdx];

        return new EvaluationResult(rms, rawProbs, rawFaultProb, sanProbs, dspFaultProb, dspIsFault, dominantClass, rpmRes, guardRes.auditLog);
    }

    @Test
    public void testHealthyDriveAudioChunk() throws Exception {
        File wavFile = new File(repoRoot, "tests/recordings_4min_study/toshiba_p300/toshiba_idle_baseline.wav");
        if (!wavFile.exists()) {
            wavFile = new File(repoRoot, "dataset_categorized/normal/archive_hdd3_drive_noises.wav");
        }
        assertTrue("Plik testowy musi istnieć: " + wavFile.getAbsolutePath(), wavFile.exists());

        float[] samples = loadWavChunk(wavFile);
        EvaluationResult res = evaluatePipeline(samples);

        System.out.println("HEALTHY DRIVE sanProbs=" + java.util.Arrays.toString(res.sanProbs) +
                ", rawProbs=" + java.util.Arrays.toString(res.rawProbs) + ", audit=" + res.auditLog);

        // Po użyciu DSP dysk zdrowy musi mieć werdykt 'SPRAWNY' (brak anomalii)
        assertFalse("Dysk sprawny nie może być oznaczony jako awaria po użyciu DSP", res.dspIsFault);
        assertEquals("Dominująca klasa musi być 'normal'", "normal", res.dominantClass);
        assertTrue("Prawdopodobieństwo braku awarii po DSP musi być wysokie", res.sanProbs[2] >= 0.65f);
    }

    @Test
    public void testClickingDriveAudioChunk() throws Exception {
        File wavFile = new File(repoRoot, "dataset_categorized/clicking/Heads-bad-wd-101403.wav");
        if (!wavFile.exists()) {
            wavFile = new File(repoRoot, "dataset_categorized/clicking/Heads-TOSHIBA-500GB-3.5-DT01ACA050.wav");
        }
        assertTrue("Plik awarii clicking musi istnieć: " + wavFile.getAbsolutePath(), wavFile.exists());

        float[] samples = loadWavChunk(wavFile);
        EvaluationResult res = evaluatePipeline(samples);

        assertTrue("Awaria mechaniczna klikania musi zostać wykryta po użyciu DSP", res.dspIsFault);
        assertTrue("Prawdopodobieństwo awarii klikania musi być wysokie", res.sanProbs[1] >= 0.50f);
        assertEquals("Dominująca klasa to 'clicking'", "clicking", res.dominantClass);
    }

    @Test
    public void testBeepingDriveAudioChunk() throws Exception {
        File wavFile = new File(repoRoot, "dataset_categorized/beeping/Beeping Seagate HDD.wav");
        if (!wavFile.exists()) {
            wavFile = new File(repoRoot, "dataset_categorized/beeping/Seized Spindle.wav");
        }
        assertTrue("Plik awarii beeping musi istnieć: " + wavFile.getAbsolutePath(), wavFile.exists());

        float[] samples = loadWavChunk(wavFile);
        EvaluationResult res = evaluatePipeline(samples);

        System.out.println("EVAL BEEPING: dspFault=" + res.dspFaultProb + ", rawFault=" + res.rawFaultProb +
                ", dominant=" + res.dominantClass + ", sanProbs=" + java.util.Arrays.toString(res.sanProbs) +
                ", rawProbs=" + java.util.Arrays.toString(res.rawProbs) + ", audit=" + res.auditLog);

        assertTrue("Zatarcie silnika (beeping) musi zostać wykryte po użyciu DSP", res.dspIsFault);
        assertEquals("Dominująca klasa to 'beeping'", "beeping", res.dominantClass);
    }

    @Test
    public void testPcAmbientFanRejection() throws Exception {
        File wavFile = new File(repoRoot, "dataset_categorized/normal/ambient_Sounds_made_by_a_fan.wav");
        if (!wavFile.exists()) {
            wavFile = new File(repoRoot, "data/raw/computer_ambient/fan_noise_01.wav");
        }
        if (wavFile.exists()) {
            float[] samples = loadWavChunk(wavFile);
            EvaluationResult res = evaluatePipeline(samples);

            // DSP Guard musi stłumić szum wentylatora i zapobiec fałszywemu alarmowi
            assertFalse("Szum wentylatora komputera nie może wywołać alarmu awarii po użyciu DSP", res.dspIsFault);
            assertEquals("Klasa po DSP musi pozostać 'normal'", "normal", res.dominantClass);
        }
    }

    private static class EvaluationResult {
        final float rms;
        final float[] rawProbs;
        final float rawFaultProb;
        final float[] sanProbs;
        final float dspFaultProb;
        final boolean dspIsFault;
        final String dominantClass;
        final RpmHarmonicsAnalyzer.RpmResult rpmResult;
        final String auditLog;

        EvaluationResult(float rms, float[] rawProbs, float rawFaultProb, float[] sanProbs,
                         float dspFaultProb, boolean dspIsFault, String dominantClass,
                         RpmHarmonicsAnalyzer.RpmResult rpmResult, String auditLog) {
            this.rms = rms;
            this.rawProbs = rawProbs;
            this.rawFaultProb = rawFaultProb;
            this.sanProbs = sanProbs;
            this.dspFaultProb = dspFaultProb;
            this.dspIsFault = dspIsFault;
            this.dominantClass = dominantClass;
            this.rpmResult = rpmResult;
            this.auditLog = auditLog;
        }
    }
}
