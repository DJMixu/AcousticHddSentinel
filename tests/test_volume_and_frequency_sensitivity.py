"""
Test wrażliwości modelu CNN na głośność (amplitudę) vs częstotliwość (spektrum).
Weryfikacja hipotezy: "możliwe że problem leży w sieci neuronowej i bazuje na głośności nie na częstotliwościach"
oraz test zachowania na dźwiękach komputera (wentylatory), tonach czystych i mowie.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.model_cnn import AcousticHddSentinelCNN
from ml.hdd_audio_encoder import encode_audio_chunk, TARGET_SR

def load_trained_model():
    model = AcousticHddSentinelCNN(num_classes=4)
    model_path = PROJECT_ROOT / "training_output" / "best_hdd_cnn_model.pt"
    cp = torch.load(model_path, map_location="cpu")
    model.load_state_dict(cp["state_dict"])
    model.eval()
    return model, cp["classes"]

def predict_audio(model, y, sr=TARGET_SR):
    tensor = encode_audio_chunk(y, sr=sr, tensor_format="pytorch")
    tensor_t = torch.from_numpy(tensor).unsqueeze(0)
    with torch.no_grad():
        probs = model(tensor_t)[0].numpy()
    return probs

def test_volume_scaling():
    print("=" * 70)
    print("EKSPERYMENT 1: Wpływ skalowania głośności (Volume Scaling) na predykcję CNN")
    print("=" * 70)
    model, classes = load_trained_model()
    
    # Wybierz po 1 próbce z każdej klasy
    sample_paths = {
        "normal": list((PROJECT_ROOT / "dataset_categorized" / "normal").glob("*.wav"))[0],
        "clicking": list((PROJECT_ROOT / "dataset_categorized" / "clicking").glob("*.*"))[0],
        "beeping": list((PROJECT_ROOT / "dataset_categorized" / "beeping").glob("*.*"))[0],
        "scratching": list((PROJECT_ROOT / "dataset_categorized" / "scratching").glob("*.*"))[0],
    }
    
    gains = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    
    for cls_name, path in sample_paths.items():
        y, sr = librosa.load(str(path), sr=TARGET_SR)
        y = y[:int(3.0 * TARGET_SR)]
        print(f"\n--- Próbka: {cls_name.upper()} ({path.name}) ---")
        base_rms = np.sqrt(np.mean(y**2))
        print(f"Oryginalny RMS: {base_rms:.4f}")
        
        for g in gains:
            y_scaled = y * g
            rms_scaled = np.sqrt(np.mean(y_scaled**2))
            probs = predict_audio(model, y_scaled, sr)
            top_idx = int(np.argmax(probs))
            top_class = classes[top_idx]
            top_p = probs[top_idx]
            
            # Orientacyjny dB SPL dla telefonu trzymanego przy dysku (skalibrowany: 1.0 = ~75 dB SPL, 0.05 = ~45 dB SPL)
            approx_db = 75.0 + 20.0 * np.log10(max(g, 1e-4))
            print(f"  Gain x{g:4.2f} (~{approx_db:4.1f} dB SPL, RMS={rms_scaled:.5f}) -> {top_class.upper():10s} (p={top_p*100:5.1f}%) | "
                  f"B:{probs[0]*100:4.1f}% C:{probs[1]*100:4.1f}% N:{probs[2]*100:4.1f}% S:{probs[3]*100:4.1f}%")

def test_frequency_tones():
    print("\n" + "=" * 70)
    print("EKSPERYMENT 2: Odpowiedź sieci na czyste częstotliwości (Pure Tones & Harmonics)")
    print("=" * 70)
    model, classes = load_trained_model()
    t = np.linspace(0, 3.0, int(3.0 * TARGET_SR), endpoint=False)
    
    test_freqs = [
        (90, "5400 RPM Fundamentalna"),
        (120, "7200 RPM Fundamentalna"),
        (240, "7200 RPM 2. Harmoniczna"),
        (360, "7200 RPM 3. Harmoniczna"),
        (720, "Silnik wrzeciona 6. harm."),
        (1000, "Typowy Beep stojana 1 kHz"),
        (1500, "Beep stojana 1.5 kHz"),
        (2500, "Pisk głowicy 2.5 kHz"),
        (4000, "Wysoki pisk / rezonans 4 kHz"),
    ]
    
    for f, desc in test_freqs:
        y_tone = 0.2 * np.sin(2 * np.pi * f * t)
        # Dodaj minimalny szum kwantyzacji przetwornika
        y_tone += 0.001 * np.random.randn(len(t))
        probs = predict_audio(model, y_tone)
        top_idx = int(np.argmax(probs))
        print(f"  Ton {f:4d} Hz ({desc:30s}) -> {classes[top_idx].upper():10s} (p={probs[top_idx]*100:5.1f}%) | "
              f"B:{probs[0]*100:4.1f}% C:{probs[1]*100:4.1f}% N:{probs[2]*100:4.1f}% S:{probs[3]*100:4.1f}%")

def test_computer_and_voice_sounds():
    print("\n" + "=" * 70)
    print("EKSPERYMENT 3: Test na dźwiękach komputera (Wentylatory PC, Szum) i mowie")
    print("=" * 70)
    model, classes = load_trained_model()
    sr = TARGET_SR
    num_samples = int(3.0 * sr)
    t = np.linspace(0, 3.0, num_samples, endpoint=False)
    
    # 1. Wentylator PC (1200 RPM = 20 Hz, 7 łopatek = 140 Hz BPF + szum aerodynamiczny)
    import scipy.signal as signal
    white = np.random.randn(num_samples)
    sos_fan = signal.butter(3, [80 / (sr/2), 2500 / (sr/2)], btype="bandpass", output="sos")
    fan_noise = signal.sosfilt(sos_fan, white) * 0.15
    fan_blade_pass = 0.08 * np.sin(2 * np.pi * 140 * t) + 0.04 * np.sin(2 * np.pi * 280 * t)
    pc_fan = (fan_noise + fan_blade_pass).astype(np.float32)
    
    # 2. Szum chłodzenia GPU / turbina (szum wyższych częstotliwości 500 - 5000 Hz)
    sos_gpu = signal.butter(4, [400 / (sr/2), 5000 / (sr/2)], btype="bandpass", output="sos")
    gpu_fan = signal.sosfilt(sos_gpu, white) * 0.18
    
    # 3. Szum tła pokoju / klimatyzacja (szum różowy / low-pass)
    sos_room = signal.butter(2, 600 / (sr/2), btype="lowpass", output="sos")
    room_noise = signal.sosfilt(sos_room, white) * 0.05
    
    # 4. Syntetyczna mowa / buczenie ustami (formanty wokalne F1=500Hz, F2=1500Hz, F3=2500Hz z wibracją strun 120Hz)
    pulse_train = signal.sawtooth(2 * np.pi * 130 * t)
    sos_f1 = signal.butter(2, [400 / (sr/2), 600 / (sr/2)], btype="bandpass", output="sos")
    sos_f2 = signal.butter(2, [1300 / (sr/2), 1700 / (sr/2)], btype="bandpass", output="sos")
    sos_f3 = signal.butter(2, [2300 / (sr/2), 2700 / (sr/2)], btype="bandpass", output="sos")
    vocal = signal.sosfilt(sos_f1, pulse_train) * 0.4 + signal.sosfilt(sos_f2, pulse_train) * 0.3 + signal.sosfilt(sos_f3, pulse_train) * 0.2
    
    # 5. Cmokanie / klikanie ustami (ostre impulsy bez szumu HDD)
    mouth_clicks = np.zeros(num_samples, dtype=np.float32)
    for c_pos in [0.5, 1.0, 1.5, 2.0, 2.5]:
        idx = int(c_pos * sr)
        click_w = int(sr * 0.015)
        mouth_clicks[idx:idx+click_w] = np.hanning(click_w) * np.sin(2 * np.pi * 1200 * np.linspace(0, 0.015, click_w))
    
    scenarios = [
        ("Wentylator PC (1200 RPM / BPF 140 Hz)", pc_fan),
        ("Szum chłodzenia GPU / turbina", gpu_fan),
        ("Szum tła pokoju (Room ambient)", room_noise),
        ("Buczenie / głos ludzki (Formanty wokalne)", vocal.astype(np.float32)),
        ("Cmokanie / klikanie ustami", mouth_clicks),
    ]
    
    for name, wave in scenarios:
        rms = np.sqrt(np.mean(wave**2))
        probs = predict_audio(model, wave)
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        print(f"  {name:42s} (RMS={rms:.4f}) -> {top_class.upper():10s} (p={probs[top_idx]*100:5.1f}%) | "
              f"B:{probs[0]*100:4.1f}% C:{probs[1]*100:4.1f}% N:{probs[2]*100:4.1f}% S:{probs[3]*100:4.1f}%")

if __name__ == "__main__":
    test_volume_scaling()
    test_frequency_tones()
    test_computer_and_voice_sounds()
