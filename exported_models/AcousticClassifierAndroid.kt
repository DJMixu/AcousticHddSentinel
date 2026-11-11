package com.acoustichddsentinel.ml

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer

/**
 * Obsługa klasyfikatora Acoustic HDD Sentinel na Androidzie przy użyciu ONNX Runtime Mobile.
 * Wymaga zależności: implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
 */
class AcousticClassifierAndroid(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    companion object {
        val CLASSES = arrayOf("beeping", "clicking", "normal", "scratching")
        const val CHANNELS = 3
        const val N_MELS = 128
        const val FRAMES = 259
        const val TOTAL_ELEMENTS = CHANNELS * N_MELS * FRAMES
    }

    init {
        // Model best_hdd_cnn_model_int8.onnx umieszczony w app/src/main/assets/
        val modelBytes = context.assets.open("best_hdd_cnn_model_int8.onnx").readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    /**
     * Wykonuje inferencję na spłaszczonym tensorze (3 * 128 * 259) floatów.
     * Zwraca parę: (Nazwa najprawdopodobniejszej klasy, Pewność w przedziale 0.0 - 1.0)
     */
    fun classifyAudioChunk(flatTensorData: FloatArray): Pair<String, Float> {
        require(flatTensorData.size == TOTAL_ELEMENTS) {
            "Tensor musi zawierać dokładnie $TOTAL_ELEMENTS próbek (3 x 128 x 259)."
        }

        val shape = longArrayOf(1, CHANNELS.toLong(), N_MELS.toLong(), FRAMES.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatTensorData), shape)

        val results = session.run(mapOf("input_audio_tensor" to tensor))
        @Suppress("UNCHECKED_CAST")
        val probabilities = (results[0].value as Array<FloatArray>)[0]

        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }

        tensor.close()
        results.close()
        return Pair(CLASSES[maxIdx], maxProb)
    }

    fun close() {
        session.close()
        env.close()
    }
}
