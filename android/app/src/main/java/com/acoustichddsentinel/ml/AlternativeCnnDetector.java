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
 * Implementacja strategii detekcji dla alternatywnego modelu CNN (SMPC / BezNormal / OOD Prototype).
 * Obsługuje lazy-loading sesji ONNX Runtime oraz bezpieczny fallback na model INT8 w razie braku pliku.
 * Zgodność z ADR-002, 100% Czysta Java 17.
 */
public class AlternativeCnnDetector implements DiskFailureDetector {

    private static final String TAG = "AlternativeCnnDetector";
    private static final String MODEL_ASSET_SMPC_INT8 = "best_smpc_model_int8.onnx";
    private static final String MODEL_ASSET_SMPC_FP32 = "best_smpc_model.onnx";
    private static final String MODEL_ASSET_FALLBACK_INT8 = "best_hdd_cnn_model_int8.onnx";
    private static final String MODEL_ASSET_FALLBACK_FP32 = "best_hdd_cnn_model.onnx";
    private static final String INPUT_NODE_NAME = "input_audio_tensor";

    public static final String[] FOUR_CLASSES = {
        "beeping",
        "clicking",
        "normal",
        "scratching"
    };

    public static final String[] THREE_FAULT_CLASSES = {
        "beeping",
        "clicking",
        "scratching"
    };

    private final Context context;
    private OrtEnvironment environment;
    private OrtSession session;
    private volatile boolean isLoaded;

    public AlternativeCnnDetector(Context context) {
        this.context = context != null ? context.getApplicationContext() : null;
    }

    @Override
    public String getModelName() {
        return ModelType.ALTERNATIVE.getTitle();
    }

    @Override
    public String getModelDescription() {
        return ModelType.ALTERNATIVE.getDescription();
    }

    @Override
    public ModelType getModelType() {
        return ModelType.ALTERNATIVE;
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

        String[] candidateAssets = new String[]{
            MODEL_ASSET_SMPC_INT8,
            MODEL_ASSET_SMPC_FP32,
            MODEL_ASSET_FALLBACK_INT8,
            MODEL_ASSET_FALLBACK_FP32
        };

        OrtSession loadedSession = null;
        String selectedAsset = null;

        for (String assetPath : candidateAssets) {
            try {
                byte[] bytes = readAssetToByteArray(context, assetPath);
                if (bytes != null && bytes.length > 0) {
                    loadedSession = environment.createSession(bytes, options);
                    selectedAsset = assetPath;
                    Log.i(TAG, "Pomyślnie załadowano alternatywny model ONNX: " + assetPath + " (" + bytes.length + " bajtów)");
                    break;
                }
            } catch (Exception ignored) {
                // Próba kolejnego kandydata z listy assetów
            }
        }

        if (loadedSession == null) {
            throw new IOException("Nie udało się załadować żadnego z alternatywnych modeli ONNX z katalogu assets.");
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

            Map<String, Float> probMap = new HashMap<>(4);
            String dominantClass;
            float maxProb;
            boolean isAnomaly;

            if (probs.length >= 4) {
                // Standardowy model 4-klasowy
                int bestIdx = 0;
                maxProb = -1.0f;
                for (int i = 0; i < FOUR_CLASSES.length; i++) {
                    float p = probs[i];
                    probMap.put(FOUR_CLASSES[i], p);
                    if (p > maxProb) {
                        maxProb = p;
                        bestIdx = i;
                    }
                }
                dominantClass = FOUR_CLASSES[bestIdx];
                isAnomaly = !"normal".equalsIgnoreCase(dominantClass);
            } else {
                // Model 3-klasowy SMPC (uszkodzenia mechaniczne)
                float pBeeping = probs.length > 0 ? probs[0] : 0.0f;
                float pClicking = probs.length > 1 ? probs[1] : 0.0f;
                float pScratching = probs.length > 2 ? probs[2] : 0.0f;
                float faultSum = pBeeping + pClicking + pScratching;
                float pNormal = Math.max(0.0f, 1.0f - faultSum);

                probMap.put("beeping", pBeeping);
                probMap.put("clicking", pClicking);
                probMap.put("normal", pNormal);
                probMap.put("scratching", pScratching);

                if (faultSum >= 0.70f) {
                    isAnomaly = true;
                    if (pBeeping >= pClicking && pBeeping >= pScratching) {
                        dominantClass = "beeping";
                        maxProb = pBeeping;
                    } else if (pClicking >= pBeeping && pClicking >= pScratching) {
                        dominantClass = "clicking";
                        maxProb = pClicking;
                    } else {
                        dominantClass = "scratching";
                        maxProb = pScratching;
                    }
                } else {
                    isAnomaly = false;
                    dominantClass = "normal";
                    maxProb = pNormal;
                }
            }

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
            Log.e(TAG, "Błąd podczas zamykania sesji ONNX w AlternativeCnnDetector", e);
        }
    }
}
