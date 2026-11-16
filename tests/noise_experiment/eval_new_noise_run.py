import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import soundfile as sf
import numpy as np
import torch
from ml.model_cnn import AcousticHddSentinelCNN
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard
from ml.rpm_harmonics_analyzer import analyze_frequency_peaks

def main():
    audio_path = "tests/noise_experiment/new_ambient_mic_noise_10s.wav"
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    model = AcousticHddSentinelCNN(num_classes=4)
    state = torch.load("training_output/best_hdd_cnn_model.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    guard = DSPAdversarialGuard()
    chunk_len = 66150
    step = chunk_len // 2
    classes = ["beeping", "clicking", "normal", "scratching"]

    print("=" * 70)
    print("TEST NOWEGO NAGRANIA SZUMU Z MIKROFONU PO AKTUALIZACJI RPM")
    print("=" * 70)

    fault_count = 0
    total_windows = 0
    for i in range(0, len(data) - chunk_len + 1, step):
        chunk = data[i:i + chunk_len].astype(np.float32)
        t = encode_audio_chunk(chunk, sr)
        t_torch = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            logits = model(t_torch)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        rpm_info = analyze_frequency_peaks(chunk, sr)
        sanitized_p, sanitized, reason = guard.protect_and_sanitize_predictions(probs, t, waveform=chunk)

        fault_p = float((sanitized_p[0] + sanitized_p[1] + sanitized_p[3]) * 100.0)
        if fault_p >= 70.0:
            fault_count += 1
        total_windows += 1

        w_idx = i // step
        t_start = i / sr
        t_end = (i + chunk_len) / sr

        san_str = ", ".join([f"{c}:{p*100:.1f}%" for c, p in zip(classes, sanitized_p)])
        rot = rpm_info["has_rotation"]
        sb = rpm_info["has_stall_beep"]
        verdict = "AWARIA" if fault_p >= 70.0 else "BEZPIECZNY"

        print(f"Okno {w_idx} ({t_start:.1f}s - {t_end:.1f}s):")
        print(f"  P(fault) po DSP: {fault_p:.1f}% -> {verdict}")
        print(f"  Rozklad: {san_str}")
        if reason:
            print(f"  Straznik: {reason}")
        print(f"  RPM: rot={rot}, stall_beep={sb}\n")

    print("-" * 70)
    print(f"Liczba okien awarii P(fault) >= 70%: {fault_count} / {total_windows}")
    final_verdict = "USZKODZONY" if fault_count >= 2 else "BEZPIECZNY / AMBIENT (SZUM TLA)"
    print(f"Werdykt ostateczny badania: {final_verdict}")
    print("=" * 70)

if __name__ == "__main__":
    main()
