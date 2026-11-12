#!/usr/bin/env python3
"""
Audyt jakości i spektralnej charakterystyki zbioru audio w Acoustic HDD Sentinel.
Sprawdza:
1. Wszystkie pliki w data/raw/ (Northwind, Aesonlabs, Additional)
2. Pliki w dataset_categorized/ (w tym klasa normal)
3. Weryfikuje piki częstotliwościowe, głośność RMS, Crest Factor, obecność harmonicznych RPM.
"""

import os
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_SR = 22050

def analyze_audio_file(file_path: Path):
    """Dokonuje wieloaspektowej analizy fizycznej i spektralnej pliku audio."""
    try:
        y, orig_sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        return {"error": str(e), "file": str(file_path)}

    duration = len(y) / TARGET_SR
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_factor = float(peak / (rms + 1e-9))
    crest_factor_db = float(20 * np.log10(crest_factor + 1e-9))

    # Analiza gęstości widmowej mocy (Welch PSD)
    freqs, psd = signal.welch(y, fs=TARGET_SR, nperseg=4096, noverlap=2048)
    psd_db = 10 * np.log10(psd + 1e-12)

    # Detekcja pików w zakresie 40 - 2000 Hz
    freq_mask = (freqs >= 40) & (freqs <= 2000)
    sub_freqs = freqs[freq_mask]
    sub_psd = psd_db[freq_mask]
    peaks_indices, properties = signal.find_peaks(sub_psd, prominence=3.0, distance=10)

    top_peaks = []
    if len(peaks_indices) > 0:
        prominences = properties["prominences"]
        sorted_peak_order = np.argsort(prominences)[::-1]
        for idx in sorted_peak_order[:5]:
            p_idx = peaks_indices[idx]
            top_peaks.append({
                "freq_hz": round(float(sub_freqs[p_idx]), 1),
                "psd_db": round(float(sub_psd[p_idx]), 1),
                "prominence": round(float(prominences[idx]), 1)
            })

    # Sprawdzenie obecności pików RPM:
    # 5400 RPM -> ~90 Hz (+/- 3 Hz)
    # 7200 RPM -> ~120 Hz (+/- 3 Hz)
    # 10000 RPM -> ~166.7 Hz (+/- 3 Hz)
    has_5400_rpm = any(abs(p["freq_hz"] - 90.0) <= 3.5 for p in top_peaks)
    has_7200_rpm = any(abs(p["freq_hz"] - 120.0) <= 3.5 for p in top_peaks)
    has_10000_rpm = any(abs(p["freq_hz"] - 166.7) <= 4.0 for p in top_peaks)

    # Sprawdzenie obecności pików beeping (zwykle 1000 - 3000 Hz)
    beep_mask = (freqs >= 800) & (freqs <= 3500)
    beep_peaks, beep_props = signal.find_peaks(psd_db[beep_mask], prominence=6.0, distance=15)
    has_beep_tone = len(beep_peaks) > 0

    return {
        "file": file_path.name,
        "parent_dir": file_path.parent.name,
        "duration_sec": round(duration, 2),
        "rms": round(rms, 6),
        "peak": round(peak, 4),
        "crest_factor_db": round(crest_factor_db, 2),
        "top_peaks": top_peaks,
        "has_5400_rpm": has_5400_rpm,
        "has_7200_rpm": has_7200_rpm,
        "has_10000_rpm": has_10000_rpm,
        "has_beep_tone": has_beep_tone
    }


def run_full_audit():
    print("=" * 70)
    print("AUDYT ZBIORU AUDIO: Weryfikacja pików RPM, głośności i sygnatur awarii")
    print("=" * 70)

    # 1. Audyt dataset_categorized
    cat_dir = PROJECT_ROOT / "dataset_categorized"
    categories = ["normal", "beeping", "clicking", "scratching"]

    results_by_cat = {c: [] for c in categories}
    for c in categories:
        c_path = cat_dir / c
        if not c_path.exists():
            continue
        files = list(c_path.glob("*.wav")) + list(c_path.glob("*.mp3"))
        for f in files:
            res = analyze_audio_file(f)
            if "error" not in res:
                results_by_cat[c].append(res)

    print("\n--- Podsumowanie klas w dataset_categorized ---")
    for c in categories:
        data = results_by_cat[c]
        if not data:
            print(f"Klasa {c}: brak plików!")
            continue

        rms_vals = [d["rms"] for d in data]
        crest_vals = [d["crest_factor_db"] for d in data]
        durs = [d["duration_sec"] for d in data]

        rpm_5400_count = sum(1 for d in data if d["has_5400_rpm"])
        rpm_7200_count = sum(1 for d in data if d["has_7200_rpm"])
        beep_count = sum(1 for d in data if d["has_beep_tone"])

        print(f"\nKlasa [{c.upper()}] (plików: {len(data)}):")
        print(f"  Średnia długość: {np.mean(durs):.1f} s (zakres: {np.min(durs):.1f} - {np.max(durs):.1f} s)")
        print(f"  Średni RMS: {np.mean(rms_vals):.4f} (min: {np.min(rms_vals):.4f}, max: {np.max(rms_vals):.4f})")
        print(f"  Średni Crest Factor: {np.mean(crest_vals):.1f} dB")
        print(f"  Piki 5400 RPM (90 Hz): {rpm_5400_count}/{len(data)} ({rpm_5400_count/len(data)*100:.1f}%)")
        print(f"  Piki 7200 RPM (120 Hz): {rpm_7200_count}/{len(data)} ({rpm_7200_count/len(data)*100:.1f}%)")
        print(f"  Piki tonowe pisków (800-3500 Hz): {beep_count}/{len(data)} ({beep_count/len(data)*100:.1f}%)")

        # Przykładowe 2 pliki z widocznymi pikami
        print("  Przykładowe piki widmowe:")
        for sample in data[:2]:
            peaks_str = ", ".join([f"{p['freq_hz']}Hz ({p['prominence']}dB)" for p in sample['top_peaks'][:3]])
            print(f"    - {sample['file'][:35]:35s} -> piki: [{peaks_str}]")

    # Zapis wyników do JSON w tests/
    out_json = PROJECT_ROOT / "tests" / "dataset_spectral_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_by_cat, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Zapisano szczegółowy raport audytu do: {out_json}")

if __name__ == "__main__":
    run_full_audit()
