"""
Test weryfikacyjny odporności systemu na głośny szum otoczenia miejskiego (City / Traffic Noise)
oraz przesterowanie mikrofonu fizycznego (Hard Clipping).

Testuje zarejestrowane nagranie: tests/noise_experiment/city_noise_recorded_from_mic.wav
Single Source of Truth: ADR-001, AGENTS.md
"""

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

TARGET_SR = 22050
CHUNK_LEN = 66150  # 3.0 s
STEP = CHUNK_LEN // 2  # 1.5 s

def test_city_noise_immunity():
    audio_path = "tests/noise_experiment/city_noise_recorded_from_mic.wav"
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    model = AcousticHddSentinelCNN(num_classes=4)
    state = torch.load("training_output/best_hdd_cnn_model.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    guard = DSPAdversarialGuard()
    classes = ["beeping", "clicking", "normal", "scratching"]

    print("=" * 70)
    print("TEST ODPORNOŚCI NA GŁOŚNY SZUM MIASTA I PRZESTEROWANIE MIKROFONU")
    print(f"Plik wejściowy: {audio_path}")
    print(f"Długość: {len(data)/sr:.2f}s, SR: {sr} Hz")
    print("=" * 70)

    fault_windows = 0
    total_windows = 0

    for i in range(0, len(data) - CHUNK_LEN + 1, STEP):
        chunk = data[i:i + CHUNK_LEN].astype(np.float32)
        t = encode_audio_chunk(chunk, sr)
        t_torch = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            logits = model(t_torch)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        rpm_info = analyze_frequency_peaks(chunk, sr)
        sanitized_p, sanitized, reason = guard.protect_and_sanitize_predictions(probs, t, waveform=chunk)

        fault_prob = (sanitized_p[0] + sanitized_p[1] + sanitized_p[3]) * 100.0
        is_fault = (fault_prob >= 70.0)

        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk**2)))
        crest = float(peak / max(rms, 1e-12))
        clip_ratio = float(np.mean(np.abs(chunk) >= 0.985 * peak))

        w_idx = i // STEP
        t_start = i / sr
        t_end = (i + CHUNK_LEN) / sr

        print(f"Okno {w_idx} ({t_start:.1f}s - {t_end:.1f}s):")
        print(f"  P(fault): {fault_prob:.1f}% -> Werdykt okna: {'AWARIA' if is_fault else 'BEZPIECZNY'}")
        print(f"  Sanityzacja ({sanitized}): {reason if reason else 'Brak interwencji'}")
        print(f"  RMS: {rms:.4f}, Crest: {crest:.2f}, Clip: {clip_ratio*100:.1f}%")
        print(f"  RPM: rot={rpm_info['has_rotation']}, stall_beep={rpm_info['has_stall_beep']}")

        if is_fault:
            fault_windows += 1
        total_windows += 1

    print("-" * 70)
    print(f"Podsumowanie: Okien z fałszywym alarmem P(fault) >= 70%: {fault_windows} / {total_windows}")
    assert fault_windows == 0, f"BŁĄD: Wykryto {fault_windows} fałszywych alarmów na szumie miasta!"
    print("STATUS: TEST ZDANY W 100% SUKCESEM (0 fałszywych alarmów)")
    print("=" * 70)

if __name__ == "__main__":
    test_city_noise_immunity()
