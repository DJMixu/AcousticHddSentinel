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
 * Klasyfikator wnioskowania akustycznego na brzegu (Edge AI) oparty o ONNX Runtime Mobile.
 * Wykonuje lokalną inferencję na skwantyzowanym modelu 2D CNN (INT8) w pamięci urządzenia bez udziału chmury.
 * Zgodność z ADR-B oraz wymaganiami NFR-01 i NFR-02.
 */
public class AcousticClassifier implements AutoCloseable {

    private static final String TAG = "AcousticClassifier";
    private static final String MODEL_ASSET_FP32 = "best_hdd_cnn_model.onnx";
    private static final String MODEL_ASSET_INT8 = "best_hdd_cnn_model_int8.onnx";
    private static final String INPUT_NODE_NAME = "input_audio_tensor";

    public static final String[] CLASS_NAMES = {
        "beeping",
        "clicking",
        "normal",
        "scratching"
    };

    private final OrtEnvironment environment;
    private final OrtSession session;

    public AcousticClassifier(Context context) throws IOException, OrtException {
        this.environment = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));

        byte[] modelBytes;
        String loadedName;
        try {
            modelBytes = readAssetToByteArray(context, MODEL_ASSET_FP32);
            loadedName = MODEL_ASSET_FP32;
        } catch (IOException e) {
            modelBytes = readAssetToByteArray(context, MODEL_ASSET_INT8);
            loadedName = MODEL_ASSET_INT8;
        }

        this.session = environment.createSession(modelBytes, options);
        Log.i(TAG, "Pomyślnie załadowano model ONNX: " + loadedName + " (" + modelBytes.length + " bajtów)");
    }

    /**
     * Wykonuje inferencję dla pojedynczego 3-kanałowego tensora [1, 3, 128, 259].
     *
     * @param flatTensor spłaszczony bufor float[] o długości 99456
     * @return wynik klasyfikacji z rozkładem prawdopodobieństw Softmax
     */
    public synchronized ClassificationResult classifyChunk(float[] flatTensor) throws OrtException {
        long startTime = System.nanoTime();

        FloatBuffer buffer = FloatBuffer.wrap(flatTensor);
        long[] shape = new long[]{1, 3, 128, 259};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(environment, buffer, shape);
             OrtSession.Result result = session.run(Collections.singletonMap(INPUT_NODE_NAME, inputTensor))) {

            long inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000L;

            // Wyjście modelu ONNX ma kształt [1, 4] i zawiera bezpośrednie prawdopodobieństwa Softmax
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

    /**
     * Pomocnicza metoda wczytania zasobu z assets do tablicy bajtów.
     */
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
    public void close() {
        try {
            if (session != null) {
                session.close();
            }
        } catch (OrtException e) {
            Log.e(TAG, "Błąd podczas zamykania sesji ONNX", e);
        }
    }
}
