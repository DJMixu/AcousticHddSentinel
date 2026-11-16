#!/usr/bin/env python3
"""
Porównanie dwóch nagrań rzeczywistego szumu mikrofonu: Take 1 vs Take 2.
Analiza fizyczna, widmowa (PSD), inferencja modelu ONNX + DspPhysicalGuard
oraz porównanie odporności po domiksowaniu do nagrań HDD.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import scipy.signal as signal
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard

TARGET_SR = 22050
CHUNK_SAMPLES = 66150  # 3.0 s
CLASSES = ["beeping", "clicking", "normal", "scratching"]


def load_audio(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32)


def compute_audio_stats(y: np.ndarray, sr: int = TARGET_SR) -> Dict:
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_factor = float(peak / max(rms, 1e-9))
    dc_offset = float(np.mean(y))

    # Analiza pasmowa
    freqs, psd = signal.welch(y, fs=sr, nperseg=4096, scaling="density")
    # Centroid widmowy
    spectral_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    # Płaskość widma (Spectral Flatness)
    geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    ari_mean = np.mean(psd) + 1e-12
    spectral_flatness = float(geo_mean / ari_mean)

    # Energia pasmowa: <1.5 kHz, 1.5-4 kHz, >4 kHz
    band_low = float(np.sum(psd[(freqs >= 50) & (freqs < 1500)]))
    band_mid = float(np.sum(psd[(freqs >= 1500) & (freqs < 4000)]))
    band_high = float(np.sum(psd[(freqs >= 4000) & (freqs < 10000)]))
    total_band = band_low + band_mid + band_high + 1e-12

    return {
        "rms": rms,
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-9))),
        "peak": peak,
        "crest_factor": crest_factor,
        "dc_offset": dc_offset,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_flatness": spectral_flatness,
        "energy_distribution": {
            "low_pct": (band_low / total_band) * 100.0,
            "mid_pct": (band_mid / total_band) * 100.0,
            "high_pct": (band_high / total_band) * 100.0,
        },
        "freqs": freqs,
        "psd": psd
    }


def evaluate_noise_with_model(noise: np.ndarray, sess, guard) -> List[Dict]:
    num_windows = len(noise) // CHUNK_SAMPLES
    results = []
    for w in range(num_windows):
        chunk = noise[w * CHUNK_SAMPLES : (w + 1) * CHUNK_SAMPLES]
        tensor = encode_audio_chunk(chunk, sr=TARGET_SR, tensor_format="pytorch")
        raw_probs = sess.run(None, {"input_audio_tensor": np.expand_dims(tensor, 0).astype(np.float32)})[0][0]
        san_probs, guarded, reason = guard.protect_and_sanitize_predictions(raw_probs, tensor, waveform=chunk)
        fault_p = float(san_probs[0] + san_probs[1] + san_probs[3])

        results.append({
            "window": w,
            "raw_probs": {CLASSES[i]: float(raw_probs[i]) for i in range(4)},
            "san_probs": {CLASSES[i]: float(san_probs[i]) for i in range(4)},
            "fault_p": fault_p,
            "is_fault_verdict": (fault_p >= 0.70),
            "guarded": guarded,
            "reason": reason
        })
    return results


def main():
    print("=" * 80)
    print("PORÓWNANIE DWÓCH NAGRAŃ SZUMU MIKROFONU: TAKE 1 vs TAKE 2")
    print("=" * 80)

    p1 = PROJECT_ROOT / "tests" / "noise_experiment" / "ambient_mic_noise_10s.wav"
    p2 = PROJECT_ROOT / "tests" / "noise_experiment" / "ambient_mic_noise_10s_take2.wav"

    if not p1.exists() or not p2.exists():
        print("[BŁĄD] Brak jednego z plików szumu!")
        return

    noise1 = load_audio(p1)
    noise2 = load_audio(p2)

    stats1 = compute_audio_stats(noise1)
    stats2 = compute_audio_stats(noise2)

    print("\n--- 1. PORÓWNANIE PARAMETRÓW SYGNAŁOWYCH ---")
    print(f"{'Parametr':<30} | {'Take 1 (Poprzedni)':<20} | {'Take 2 (Nowy)':<20} | {'Różnica':<15}")
    print("-" * 90)
    print(f"{'Długość nagrania':<30} | {len(noise1)/TARGET_SR:<18.2f} s | {len(noise2)/TARGET_SR:<18.2f} s | {(len(noise2)-len(noise1))/TARGET_SR:+.2f} s")
    print(f"{'Wartość skuteczna (RMS)':<30} | {stats1['rms']:<20.6f} | {stats2['rms']:<20.6f} | {stats2['rms']-stats1['rms']:+.6f}")
    print(f"{'Poziom w dBFS':<30} | {stats1['rms_dbfs']:<18.2f} dB | {stats2['rms_dbfs']:<18.2f} dB | {stats2['rms_dbfs']-stats1['rms_dbfs']:+.2f} dB")
    print(f"{'Wartość szczytowa (Peak)':<30} | {stats1['peak']:<20.6f} | {stats2['peak']:<20.6f} | {stats2['peak']-stats1['peak']:+.6f}")
    print(f"{'Współczynnik Crest Factor':<30} | {stats1['crest_factor']:<20.2f} | {stats2['crest_factor']:<20.2f} | {stats2['crest_factor']-stats1['crest_factor']:+.2f}")
    print(f"{'DC Offset':<30} | {stats1['dc_offset']:<20.6f} | {stats2['dc_offset']:<20.6f} | {stats2['dc_offset']-stats1['dc_offset']:+.6f}")
    print(f"{'Centroid widmowy':<30} | {stats1['spectral_centroid_hz']:<18.1f} Hz | {stats2['spectral_centroid_hz']:<18.1f} Hz | {stats2['spectral_centroid_hz']-stats1['spectral_centroid_hz']:+.1f} Hz")
    print(f"{'Płaskość widma (Flatness)':<30} | {stats1['spectral_flatness']:<20.4f} | {stats2['spectral_flatness']:<20.4f} | {stats2['spectral_flatness']-stats1['spectral_flatness']:+.4f}")
    print("-" * 90)
    print("Rozkład energii pasmowej:")
    print(f"  * Pasmo niskie (50 - 1500 Hz):   Take 1: {stats1['energy_distribution']['low_pct']:5.1f}%  vs  Take 2: {stats2['energy_distribution']['low_pct']:5.1f}%")
    print(f"  * Pasmo średnie (1.5 - 4 kHz):   Take 1: {stats1['energy_distribution']['mid_pct']:5.1f}%  vs  Take 2: {stats2['energy_distribution']['mid_pct']:5.1f}%")
    print(f"  * Pasmo wysokie (4 - 10 kHz):    Take 1: {stats1['energy_distribution']['high_pct']:5.1f}%  vs  Take 2: {stats2['energy_distribution']['high_pct']:5.1f}%")

    # Korelacja widmowa między dwoma nagraniami
    psd_corr = np.corrcoef(stats1["psd"], stats2["psd"])[0, 1]
    print(f"\nWspółczynnik korelacji widmowej Pearsona (PSD): r = {psd_corr:.4f} (bliski 1.0 oznacza identyczny profil tła)")

    # 2. Ewaluacja przez model ONNX i DspPhysicalGuard
    print("\n--- 2. PORÓWNANIE WERDYKTU MODELU ONNX & STRAŻNIKA DSP ---")
    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    guard = DSPAdversarialGuard()

    eval1 = evaluate_noise_with_model(noise1, sess, guard)
    eval2 = evaluate_noise_with_model(noise2, sess, guard)

    print(f"{'Okno':<6} | {'Take 1: P(fault)':<18} | {'Take 1: Klasa dominant':<22} | {'Take 2: P(fault)':<18} | {'Take 2: Klasa dominant':<22}")
    print("-" * 92)
    max_w = min(len(eval1), len(eval2))
    for w in range(max_w):
        r1 = eval1[w]
        r2 = eval2[w]
        dom1 = max(r1["san_probs"], key=r1["san_probs"].get)
        dom2 = max(r2["san_probs"], key=r2["san_probs"].get)
        print(f"Okno {w:<1} | {r1['fault_p']*100:5.1f}% (weto: {r1['guarded']}) | {dom1} ({r1['san_probs'][dom1]*100:.1f}%)        | {r2['fault_p']*100:5.1f}% (weto: {r2['guarded']}) | {dom2} ({r2['san_probs'][dom2]*100:.1f}%)")

    faults1 = sum(1 for r in eval1 if r["is_fault_verdict"])
    faults2 = sum(1 for r in eval2 if r["is_fault_verdict"])
    print(f"\nPodsumowanie werdyktu na szumie:")
    print(f"  - Take 1: {faults1}/{len(eval1)} fałszywych alarmów (Werdykt: NIE WIADOMO / SZUM TŁA)")
    print(f"  - Take 2: {faults2}/{len(eval2)} fałszywych alarmów (Werdykt: NIE WIADOMO / SZUM TŁA)")

    # 3. Wykres diagnostyczny porównania widma
    output_dir = PROJECT_ROOT / "tests" / "noise_experiment"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    # Przebieg czasowy
    t1 = np.linspace(0, len(noise1)/TARGET_SR, len(noise1), endpoint=False)
    t2 = np.linspace(0, len(noise2)/TARGET_SR, len(noise2), endpoint=False)
    ax1.plot(t1[:TARGET_SR*3], noise1[:TARGET_SR*3], label="Take 1 (Pierwsze nagranie)", color="#3B82F6", alpha=0.7, linewidth=0.8)
    ax1.plot(t2[:TARGET_SR*3], noise2[:TARGET_SR*3], label="Take 2 (Nowe nagranie)", color="#EF4444", alpha=0.7, linewidth=0.8)
    ax1.set_title("Przebieg czasowy fali szumu otoczenia (pierwsze 3 sekundy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Czas (s)")
    ax1.set_ylabel("Amplituda znormalizowana")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Gęstość mocy widmowej (PSD)
    freqs = stats1["freqs"]
    psd1_db = 10 * np.log10(stats1["psd"] + 1e-12)
    psd2_db = 10 * np.log10(stats2["psd"] + 1e-12)
    ax2.plot(freqs, psd1_db, label=f"Take 1 (RMS: {stats1['rms']:.6f}, Centroid: {stats1['spectral_centroid_hz']:.0f} Hz)", color="#3B82F6", linewidth=1.2)
    ax2.plot(freqs, psd2_db, label=f"Take 2 (RMS: {stats2['rms']:.6f}, Centroid: {stats2['spectral_centroid_hz']:.0f} Hz)", color="#EF4444", linewidth=1.2, linestyle="--")
    ax2.set_title("Gęstość Mocy Widmowej Welch PSD (dB/Hz) - Profil akustyczny tła", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Częstotliwość (Hz)")
    ax2.set_ylabel("Moc (dB/Hz)")
    ax2.set_xlim(50, 10000)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    chart_path = output_dir / "noise_take1_vs_take2_comparison.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"\n[Wykres] Zapisano wykres porównawczy: {chart_path}")

    # Zapis raportu do JSON
    report_data = {
        "take1_stats": {k: v for k, v in stats1.items() if k not in ["freqs", "psd"]},
        "take2_stats": {k: v for k, v in stats2.items() if k not in ["freqs", "psd"]},
        "psd_correlation": float(psd_corr),
        "take1_eval": eval1,
        "take2_eval": eval2
    }
    with open(output_dir / "noise_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[Raport] Zapisano raport numeryczny: {output_dir / 'noise_comparison_report.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
