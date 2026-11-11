package com.acoustichddsentinel.ml

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import com.acoustichddsentinel.dsp.DspConstants
import java.nio.FloatBuffer

/**
 * Silnik inferencyjny modelu Acoustic HDD Sentinel w technologii ONNX Runtime Mobile.
 * Wykorzystuje lekki, skwantyzowany model INT8 (1.17 MB) z katalogu assets.
 */
class AcousticClassifier(context: Context, modelAssetPath: String = "best_hdd_cnn_model_int8.onnx") {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    companion object {
        const val INPUT_NODE_NAME = "input_audio_tensor"
        val CLASSES = DspConstants.CLASSES
    }

    init {
        val modelBytes = context.assets.open(modelAssetPath).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, sessionOptions)
    }

    /**
     * Klasyfikuje przetworzony tensor audio (spłaszczony bufor 3 x 128 x 259).
     */
    fun classify(flatTensorData: FloatArray): ClassificationResult {
        require(flatTensorData.size == DspConstants.TOTAL_TENSOR_ELEMENTS) {
            "Nieprawidłowy rozmiar bufora: oczekiwano ${DspConstants.TOTAL_TENSOR_ELEMENTS}, otrzymano ${flatTensorData.size}"
        }

        val startTime = System.currentTimeMillis()

        // Kształt tensora: (batch_size=1, channels=3, n_mels=128, frames=259)
        val shape = longArrayOf(1, DspConstants.CHANNELS.toLong(), DspConstants.N_MELS.toLong(), DspConstants.FRAMES.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatTensorData), shape)

        val results = session.run(mapOf(INPUT_NODE_NAME to tensor))
        @Suppress("UNCHECKED_CAST")
        val rawProbabilities = (results[0].value as Array<FloatArray>)[0]

        tensor.close()
        results.close()

        val inferenceTimeMs = System.currentTimeMillis() - startTime

        // Mapowanie prawdopodobieństw na nazwy klas
        val probMap = mutableMapOf<String, Float>()
        var bestIdx = 0
        var bestProb = -1.0f

        for (i in CLASSES.indices) {
            val prob = if (i < rawProbabilities.size) rawProbabilities[i] else 0.0f
            val className = CLASSES[i]
            probMap[className] = prob
            if (prob > bestProb) {
                bestProb = prob
                bestIdx = i
            }
        }

        val predictedClass = CLASSES[bestIdx]
        val isAnomaly = !predictedClass.equals("normal", ignoreCase = true)

        return ClassificationResult(
            predictedClass = predictedClass,
            confidence = bestProb,
            probabilities = probMap,
            isAnomaly = isAnomaly,
            inferenceTimeMs = inferenceTimeMs
        )
    }

    fun close() {
        session.close()
        env.close()
    }
}
