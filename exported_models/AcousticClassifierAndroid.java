package com.acoustichddsentinel.ml;

import android.content.Context;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.Collections;

/**
 * Szablon inferencji Acoustic HDD Sentinel w 100% czystej Javie 17 (ADR-002).
 * Model ONNX INT8: best_hdd_cnn_model_int8.onnx (1.18 MB) w app/src/main/assets/.
 */
public class AcousticClassifierAndroid implements AutoCloseable {
    public static final String[] CLASSES = {"beeping", "clicking", "normal", "scratching"};
    public static final int CHANNELS = 3;
    public static final int N_MELS = 128;
    public static final int FRAMES = 259;
    public static final int TOTAL_ELEMENTS = CHANNELS * N_MELS * FRAMES;

    private final OrtEnvironment env;
    private OrtSession session;
    private volatile boolean isClosed;

    public AcousticClassifierAndroid(Context context) throws IOException, OrtException {
        this.env = OrtEnvironment.getEnvironment();
        byte[] modelBytes = readAssetBytes(context, "best_hdd_cnn_model_int8.onnx");
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        this.session = env.createSession(modelBytes, options);
    }

    public synchronized ClassificationResult classifyAudioChunk(float[] flatTensorData) throws OrtException {
        if (flatTensorData == null || flatTensorData.length != TOTAL_ELEMENTS) {
            throw new IllegalArgumentException("Tensor musi zawierać dokładnie " + TOTAL_ELEMENTS + " elementów.");
        }
        long[] shape = new long[]{1, CHANNELS, N_MELS, FRAMES};
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatTensorData), shape);
             OrtSession.Result results = session.run(Collections.singletonMap("input_audio_tensor", tensor))) {
            float[][] probs = (float[][]) results.get(0).getValue();
            int maxIdx = 0;
            float maxProb = probs[0][0];
            for (int i = 1; i < probs[0].length; i++) {
                if (probs[0][i] > maxProb) {
                    maxProb = probs[0][i];
                    maxIdx = i;
                }
            }
            return new ClassificationResult(CLASSES[maxIdx], maxProb);
        }
    }

    private static byte[] readAssetBytes(Context context, String assetName) throws IOException {
        try (InputStream is = context.getAssets().open(assetName);
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int r;
            while ((r = is.read(buf)) != -1) {
                baos.write(buf, 0, r);
            }
            return baos.toByteArray();
        }
    }

    @Override
    public synchronized void close() {
        if (isClosed) return;
        isClosed = true;
        try {
            if (session != null) {
                session.close();
                session = null;
            }
        } catch (OrtException ignored) {}
    }

    public static class ClassificationResult {
        public final String predictedClass;
        public final float confidence;
        public ClassificationResult(String predictedClass, float confidence) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
        }
    }
}
