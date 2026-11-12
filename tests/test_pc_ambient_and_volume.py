#!/usr/bin/env python3
"""
Test weryfikacyjny: Czy sieć neuronowa klasyfikuje normalne dźwięki komputera
(szum wentylatorów, chłodzenie, 45 dB) jako awarię dysku?
Analizuje zależność między głośnością (amplitudą) a częstotliwościami.
"""

import os
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_cnn import AcousticHddSentinelCNN
from ml.hdd_audio_encoder import encode_audio_chunk, TARGET_SR, CHUNK_DURATION

CLASS_NAMES = ["beeping", "clicking", "normal", "scratching"]


def load_trained_model():
    model = AcousticHddSentinelCNN(num_classes=4)
    model_path = PROJECT_ROOT / "training_output" / "best_hdd_cnn_model.pt"
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def generate_pc_ambient_fan_noise(duration_sec=3.0, sr=TARGET_SR, rms_level=0.005):
    """
    Syntetyzuje realistyczny szum tła komputera (~45 dB SPL):
    - Wentylator obudowy 120mm (1200 RPM, 7 łopatek -> BPF = 140 Hz)
    - Wentylator CPU 92mm (1600 RPM, 9 łopatek -> BPF = 240 Hz)
    - Szum turbulentny przepływu powietrza (filtr różowy 100 - 3000 Hz)
    - Rezonans obudowy blaszanej (niski buczący ton 50 Hz / 100 Hz od transformatora zasilacza)
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Przepływ powietrza (różowy szum filtrowany pasmowo)
    white = np.random.randn(n_samples)
    sos_air = signal.butter(3, [80 / (sr / 2), 2500 / (sr / 2)], btype="bandpass", output="sos")
    air_noise = signal.sosfilt(sos_air, white)

    # 2. Składowe wentylatorów (BPF) z lekkim flutterem
    fan_case = 0.20 * np.sin(2 * np.pi * 140.0 * t + np.random.uniform(0, 2 * np.pi))
    fan_cpu = 0.15 * np.sin(2 * np.pi * 240.0 * t + np.random.uniform(0, 2 * np.pi))

    # 3. Przydźwięk zasilacza / transformatora (50 Hz / 100 Hz hum)
    psu_hum = 0.10 * np.sin(2 * np.pi * 100.0 * t) + 0.05 * np.sin(2 * np.pi * 50.0 * t)

    combined = air_noise + fan_case + fan_cpu + psu_hum

    # Skalowanie do pożądanego RMS (~45 dB w warunkach smartfona)
    current_rms = np.sqrt(np.mean(combined ** 2))
    if current_rms > 1e-9:
        combined = combined * (rms_level / current_rms)

    return combined.astype(np.float32)


def generate_healthy_drive_in_pc(rpm=7200, duration_sec=3.0, sr=TARGET_SR, pc_rms=0.005, hdd_ratio=0.5):
    """
    Syntetyzuje zdrowy dysk HDD pracujący wewnątrz obudowy PC:
    - Szum wentylatorów PC (tło)
    - Rotacja wrzeciona HDD (90 Hz dla 5400 RPM lub 120 Hz dla 7200 RPM)
    - Szum łożysk FDB i aerodynamiki talerzy
    """
    pc_noise = generate_pc_ambient_fan_noise(duration_sec, sr, rms_level=pc_rms)

    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    f_rot = rpm / 60.0  # 90 Hz lub 120 Hz

    # Piki harmoniczne rotacji talerzy
    motor = 0.40 * np.sin(2 * np.pi * f_rot * t)
    motor += 0.25 * np.sin(2 * np.pi * (2 * f_rot) * t)
    motor += 0.15 * np.sin(2 * np.pi * (3 * f_rot) * t)
    motor += 0.10 * np.sin(2 * np.pi * (4 * f_rot) * t)

    # Szum łożysk FDB
    fdb_noise = signal.sosfilt(
        signal.butter(2, [300 / (sr / 2), 4000 / (sr / 2)], btype="bandpass", output="sos"),
        np.random.randn(n_samples)
    ) * 0.30

    hdd_signal = motor + fdb_noise
    hdd_rms = pc_rms * hdd_ratio
    hdd_signal = hdd_signal * (hdd_rms / (np.sqrt(np.mean(hdd_signal ** 2)) + 1e-9))

    return (pc_noise + hdd_signal).astype(np.float32)


def test_pc_sounds():
    print("=" * 70)
    print("TEST: Reakcja sieci CNN na dźwięki tła komputera (wentylatory, 45 dB)")
    print("=" * 70)

    model = load_trained_model()

    # Różne poziomy głośności (RMS):
    # 0.003 (~40 dB - bardzo cichy PC)
    # 0.005 (~45 dB - standardowy szum komputera z dyskiem)
    # 0.015 (~55 dB - głośniejsze chłodzenie / obciążenie)
    # 0.050 (~70 dB - głośny szum / bliski mikrofon)
    rms_levels = [0.003, 0.005, 0.015, 0.050]

    print("\n1. Test czystego szumu komputera (wentylatory, zasilacz, brak uszkodzeń):")
    print("-" * 70)
    for rms in rms_levels:
        db_approx = 45.0 + 20 * np.log10(rms / 0.005)
        audio = generate_pc_ambient_fan_noise(rms_level=rms)

        # Enkodowanie do tensora (Channel-First dla PyTorch)
        tensor = encode_audio_chunk(audio, sr=TARGET_SR, tensor_format="pytorch")
        tensor_t = torch.from_numpy(tensor).unsqueeze(0)

        with torch.no_grad():
            probs = model(tensor_t)[0].numpy()

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx]

        status = "BŁĄD (Fałszywa awaria)" if pred_class != "normal" else "POPRAWNIE (Normal)"
        probs_str = " | ".join([f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4)])
        print(f"RMS: {rms:.4f} (~{db_approx:.0f} dB) -> Wykryto: [{pred_class.upper():10s}] ({conf*100:.1f}%) -> {status}")
        print(f"       Rozkład: [{probs_str}]")

    print("\n2. Test zdrowego dysku 5400 RPM wewnątrz komputera:")
    print("-" * 70)
    for rms in [0.003, 0.005, 0.015]:
        audio = generate_healthy_drive_in_pc(rpm=5400, pc_rms=rms)
        tensor = encode_audio_chunk(audio, sr=TARGET_SR, tensor_format="pytorch")
        tensor_t = torch.from_numpy(tensor).unsqueeze(0)

        with torch.no_grad():
            probs = model(tensor_t)[0].numpy()

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx]
        status = "BŁĄD (Fałszywa awaria)" if pred_class != "normal" else "POPRAWNIE (Normal)"
        probs_str = " | ".join([f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4)])
        print(f"5400 RPM (RMS {rms:.4f}) -> Wykryto: [{pred_class.upper():10s}] ({conf*100:.1f}%) -> {status}")
        print(f"       Rozkład: [{probs_str}]")

    print("\n3. Test zdrowego dysku 7200 RPM wewnątrz komputera:")
    print("-" * 70)
    for rms in [0.003, 0.005, 0.015]:
        audio = generate_healthy_drive_in_pc(rpm=7200, pc_rms=rms)
        tensor = encode_audio_chunk(audio, sr=TARGET_SR, tensor_format="pytorch")
        tensor_t = torch.from_numpy(tensor).unsqueeze(0)

        with torch.no_grad():
            probs = model(tensor_t)[0].numpy()

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx]
        status = "BŁĄD (Fałszywa awaria)" if pred_class != "normal" else "POPRAWNIE (Normal)"
        probs_str = " | ".join([f"{CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4)])
        print(f"7200 RPM (RMS {rms:.4f}) -> Wykryto: [{pred_class.upper():10s}] ({conf*100:.1f}%) -> {status}")
        print(f"       Rozkład: [{probs_str}]")


if __name__ == "__main__":
    test_pc_sounds()
