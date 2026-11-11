package com.acoustichddsentinel.dsp

/**
 * Niezmienniki techniczne Digital Signal Processing (DSP) zgodne z ADR-001.
 */
object DspConstants {
    const val SAMPLE_RATE = 22050
    const val CHUNK_DURATION_SEC = 3.0f
    const val CHUNK_SAMPLES = (SAMPLE_RATE * CHUNK_DURATION_SEC).toInt() // 66150 próbek

    const val N_FFT = 1024
    const val HOP_LENGTH = 256
    const val N_MELS = 128
    const val FMIN = 50.0f
    const val FMAX = 10000.0f

    // Wyjściowe wymiary tensora w pamięci
    const val CHANNELS = 3
    const val FRAMES = 259
    const val TOTAL_TENSOR_ELEMENTS = CHANNELS * N_MELS * FRAMES // 3 * 128 * 259 = 99456

    // Próg odrzucania ciszy i szumu tła (RMS)
    const val RMS_SILENCE_THRESHOLD = 0.002f

    // Nazwy klas zgodne z modelem
    val CLASSES = arrayOf("beeping", "clicking", "normal", "scratching")
}
