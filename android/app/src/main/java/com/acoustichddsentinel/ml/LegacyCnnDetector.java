package com.acoustichddsentinel.ml;

import android.content.Context;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Implementacja strategii detekcji dla klasycznego 4-klasowego modelu 2D CNN.
 * Obsługuje lazy-loading sesji ONNX Runtime oraz dynamiczne zwalnianie zasobów.
 * Zgodność z ADR-002, 100% Czysta Java 17.
 */
public class LegacyCnnDetector implements DiskFailureDetector {

    private static final String TAG = "LegacyCnnDetector";
    private static final String MODEL_ASSET_INT8 = "best_hdd_cnn_model_int8.onnx";
    private static final String MODEL_ASSET_FP32 = "best_hdd_cnn_model.onnx";
    private static final String INPUT_NODE_NAME = "input_audio_tensor";

    public static final String[] CLASS_NAMES = {
        "beeping",
        "clicking",
        "normal",
        "scratching"
    };

    private final Context context;
    private OrtEnvironment environment;
    private OrtSession session;
    private volatile boolean isLoaded;

    public LegacyCnnDetector(Context context) {
        this.context = context != null ? context.getApplicationContext() : null;
    }

    @Override
    public String getModelName() {
        return ModelType.LEGACY.getTitle();
    }

    @Override
    public String getModelDescription() {
        return ModelType.LEGACY.getDescription();
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LEGACY;
    }

    @Override
    public synchronized boolean isLoaded() {
        return isLoaded && session != null;
    }

    @Override
    public synchronized void load() throws IOException, OrtException {
        if (isLoaded && session != null) {
            return;
        }

        if (context == null) {
            throw new IllegalStateException("Kontekst aplikacji jest null - nie można załadować modelu z assets.");
        }

        if (this.environment == null) {
            this.environment = OrtEnvironment.getEnvironment();
        }

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));

        byte[] modelBytes;
        OrtSession loadedSession = null;
        String loadedName;

        try {
            modelBytes = readAssetToByteArray(context, MODEL_ASSET_INT8);
            loadedSession = environment.createSession(modelBytes, options);
            loadedName = MODEL_ASSET_INT8;
            Log.i(TAG, "Pomyślnie załadowano klasyczny model ONNX INT8: " + loadedName + " (" + modelBytes.length + " bajtów)");
        } catch (Exception e) {
            Log.w(TAG, "Nie udało się uruchomić sesji INT8 (" + e.getMessage() + "), automatyczny fallback na bazowy FP32...");
            modelBytes = readAssetToByteArray(context, MODEL_ASSET_FP32);
            loadedSession = environment.createSession(modelBytes, options);
            loadedName = MODEL_ASSET_FP32;
            Log.i(TAG, "Pomyślnie załadowano klasyczny model ONNX FP32: " + loadedName + " (" + modelBytes.length + " bajtów)");
        }

        this.session = loadedSession;
        this.isLoaded = true;
    }

    @Override
    public synchronized ClassificationResult classifyChunk(float[] flatTensor) throws Exception {
        if (!isLoaded()) {
            load();
        }

        long startTime = System.nanoTime();

        FloatBuffer buffer = FloatBuffer.wrap(flatTensor);
        long[] shape = new long[]{1, 3, 128, 259};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, buffer, shape);
             OrtSession.Result result = session.run(Collections.singletonMap(INPUT_NODE_NAME, inputTensor))) {

            long inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000L;

            float[][] outputProbabilities = (float[][]) result.get(0).getValue();
            float[] probs = outputProbabilities[0];

            int bestIdx = 0;
            float maxProb = -1.0f;
            Map<String, Float> probMap = new HashMap<>(CLASS_NAMES.length);

            for (int i = 0; i < CLASS_NAMES.length; i++) {
                float p = probs[i];
                probMap.put(CLASS_NAMES[i], p);
                if (p > maxProb) {
                    maxProb = p;
                    bestIdx = i;
                }
            }

            String dominantClass = CLASS_NAMES[bestIdx];
            boolean isAnomaly = !"normal".equalsIgnoreCase(dominantClass);

            return new ClassificationResult(
                dominantClass,
                maxProb,
                probMap,
                isAnomaly,
                inferenceTimeMs
            );
        }
    }

    private static byte[] readAssetToByteArray(Context context, String assetPath) throws IOException {
        try (InputStream is = context.getAssets().open(assetPath);
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int read;
            while ((read = is.read(buf)) != -1) {
                baos.write(buf, 0, read);
            }
            return baos.toByteArray();
        }
    }

    @Override
    public synchronized void close() {
        if (!isLoaded && session == null) {
            return;
        }
        isLoaded = false;
        try {
            if (session != null) {
                session.close();
                session = null;
                Log.i(TAG, "Pomyślnie zamknięto sesję ONNX dla: " + getModelName());
            }
        } catch (OrtException e) {
            Log.e(TAG, "Błąd podczas zamykania sesji ONNX w LegacyCnnDetector", e);
        }
    }
}
