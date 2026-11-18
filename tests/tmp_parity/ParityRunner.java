
package com.acoustichddsentinel.dsp;

import java.io.*;
import java.nio.*;

public class ParityRunner {
    public static void main(String[] args) throws Exception {
        File inFile = new File("/home/DJMixu/Pulpit/AcousticHddSentinel/tests/tmp_parity/test_signal_in.raw");
        File outFile = new File("/home/DJMixu/Pulpit/AcousticHddSentinel/tests/tmp_parity/java_tensor_out.raw");

        byte[] inBytes = new byte[(int) inFile.length()];
        try (FileInputStream fis = new FileInputStream(inFile)) {
            fis.read(inBytes);
        }

        FloatBuffer fb = ByteBuffer.wrap(inBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] samples = new float[fb.remaining()];
        fb.get(samples);

        AudioPreprocessor preprocessor = new AudioPreprocessor();
        float[] flatTensor = preprocessor.processAudioChunkToFlatTensor(samples);

        ByteBuffer outBb = ByteBuffer.allocate(flatTensor.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : flatTensor) {
            outBb.putFloat(v);
        }

        try (FileOutputStream fos = new FileOutputStream(outFile)) {
            fos.write(outBb.array());
        }
    }
}
