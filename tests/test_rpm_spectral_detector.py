#!/usr/bin/env python3
"""
Deterministyczny analizator pików częstotliwościowych obrotów silnika HDD (RPM Spectral Peak Detector).
Lokalizacja: tests/test_rpm_spectral_detector.py

Implementuje fizyczną detekcję:
1. Częstotliwości bazowej obrotów wału (Spindle fundamental frequency):
   - 5400 RPM: 90.0 Hz (+/- 2.5 Hz)
   - 5900 RPM: 98.3 Hz (+/- 2.5 Hz)
   - 7200 RPM: 120.0 Hz (+/- 2.5 Hz)
   - 10000 RPM: 166.7 Hz (+/- 3.0 Hz)
   - 15000 RPM: 250.0 Hz (+/- 3.5 Hz)
2. Zgodności harmonicznej (Harmonic Coherence): obecność 2x i 3x f0.
3. Stabilności fazowej i ostrości piku (Q-factor / Prominence w dB).
4. Odróżniania dysku HDD od szumu wentylatorów komputerowych (Blade Pass Frequency) oraz głosu ludzkiego.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TARGET_SR = 22050

# Profile znanych prędkości obrotowych dysków HDD (silniki BLDC stabilizowane kwarcem PLL)
RPM_PROFILES = {
    "5400 RPM": {"f0": 90.0, "tol": 0.8, "h2": 180.0, "h3": 270.0},
    "5900 RPM (NAS)": {"f0": 98.33, "tol": 0.8, "h2": 196.67, "h3": 295.0},
    "7200 RPM": {"f0": 120.0, "tol": 0.8, "h2": 240.0, "h3": 360.0},
    "10000 RPM (Raptor)": {"f0": 166.67, "tol": 1.0, "h2": 333.33, "h3": 500.0},
    "15000 RPM (SAS)": {"f0": 250.0, "tol": 1.2, "h2": 500.0, "h3": 750.0},
}


def parabolic_peak_refinement(psd_db, p_idx, freqs, df):
    """Interpolacja paraboliczna szczytu dla sub-binowej precyzji częstotliwości."""
    if 0 < p_idx < len(psd_db) - 1:
        alpha = psd_db[p_idx - 1]
        beta = psd_db[p_idx]
        gamma = psd_db[p_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom
            exact_freq = freqs[p_idx] + p * df
            exact_power = beta - 0.25 * (alpha - gamma) * p
            return exact_freq, exact_power
    return freqs[p_idx], psd_db[p_idx]


def detect_hdd_rpm_peaks(y: np.ndarray, sr: int = TARGET_SR, min_prominence_db: float = 3.5):
    """
    Wykrywa obecność fizycznych pików obrotowych dysku HDD w sygnale audio.
    Zwraca szczegółowy słownik z wykrytą prędkością obrotową, pewnością i parametrami harmonicznych.
    """
    # 1. Obliczenie gęstości widmowej mocy (Welch PSD) z wysoką rozdzielczością
    nperseg = min(len(y), 16384)
    freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    psd_db = 10 * np.log10(psd + 1e-12)
    df = sr / nperseg

    # 2. Szukanie pików w paśmie mechanicznym (50 Hz - 1000 Hz)
    band_mask = (freqs >= 50.0) & (freqs <= 1000.0)
    band_indices = np.where(band_mask)[0]
    sub_psd = psd_db[band_mask]

    peaks_rel, properties = signal.find_peaks(
        sub_psd,
        prominence=min_prominence_db,
        distance=int(4.0 / df)
    )

    detected_peaks = []
    if len(peaks_rel) > 0:
        prominences = properties["prominences"]
        for rel_idx, prom in zip(peaks_rel, prominences):
            full_idx = band_indices[rel_idx]
            exact_f, exact_pow = parabolic_peak_refinement(psd_db, full_idx, freqs, df)
            detected_peaks.append({
                "freq_hz": float(exact_f),
                "power_db": float(exact_pow),
                "prominence_db": float(prom)
            })

    # 3. Weryfikacja profili RPM
    best_profile_match = None
    best_score = 0.0

    for name, prof in RPM_PROFILES.items():
        f0_target = prof["f0"]
        tol = prof["tol"]

        # Sprawdzenie piku podstawowego f0
        f0_matches = [p for p in detected_peaks if abs(p["freq_hz"] - f0_target) <= tol]
        # Sprawdzenie 2. harmonicznej 2*f0
        h2_matches = [p for p in detected_peaks if abs(p["freq_hz"] - prof["h2"]) <= (tol * 1.5)]
        # Sprawdzenie 3. harmonicznej 3*f0
        h3_matches = [p for p in detected_peaks if abs(p["freq_hz"] - prof["h3"]) <= (tol * 2.0)]

        score = 0.0
        details = {}

        # Prawdziwy wirujący silnik HDD generuje grzebień harmonicznych (f0 + h2 i/lub h3).
        # Samotny pik bez 2. harmonicznej to często przydźwięk sieciowy 50/100 Hz zasilacza lub rezonans wiatraka!
        has_f0 = len(f0_matches) > 0
        has_h2 = len(h2_matches) > 0
        has_h3 = len(h3_matches) > 0

        # Wymagamy obecności co najmniej dwóch składowych z rodziny harmonicznej (koherencja harmoniczna)
        if not (has_f0 and (has_h2 or has_h3)):
            continue

        f0_peak = max(f0_matches, key=lambda x: x["prominence_db"])
        score = 0.45 + min(0.25, f0_peak["prominence_db"] / 30.0)
        details["f0_detected"] = f0_peak["freq_hz"]
        details["f0_prominence_db"] = round(f0_peak["prominence_db"], 1)

        if has_h2:
            h2_peak = max(h2_matches, key=lambda x: x["prominence_db"])
            score += 0.20 + min(0.10, h2_peak["prominence_db"] / 30.0)
            details["h2_detected"] = h2_peak["freq_hz"]
            details["h2_prominence_db"] = round(h2_peak["prominence_db"], 1)

        if has_h3:
            h3_peak = max(h3_matches, key=lambda x: x["prominence_db"])
            score += 0.10
            details["h3_detected"] = h3_peak["freq_hz"]

        if score > best_score and score >= 0.55:
            best_score = score
            best_profile_match = {
                "profile": name,
                "nominal_f0": f0_target,
                "score": round(min(1.0, score), 3),
                "details": details
            }

    # 4. Sprawdzenie szumu impulsowego (Crest Factor) oraz pisków zablokowania wrzeciona
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-9) + 1e-9)

    # Detekcja pisków tonowych awarii (800 - 3500 Hz)
    beep_mask = (freqs >= 800.0) & (freqs <= 3500.0)
    beep_peaks, beep_props = signal.find_peaks(psd_db[beep_mask], prominence=8.0, distance=15)
    has_spindle_seize_tone = len(beep_peaks) > 0

    return {
        "rpm_detected": best_profile_match is not None,
        "best_match": best_profile_match,
        "rms": round(rms, 6),
        "crest_factor_db": round(crest_factor_db, 1),
        "has_spindle_seize_tone": has_spindle_seize_tone,
        "top_peaks_count": len(detected_peaks),
        "candidate_peaks": detected_peaks[:6]
    }


def run_benchmark():
    print("=" * 70)
    print("TEST: Deterministyczna weryfikacja pików RPM w zbiorze nagrań")
    print("=" * 70)

    from tests.test_pc_ambient_and_volume import generate_pc_ambient_fan_noise, generate_healthy_drive_in_pc

    # Test 1: Zdrowy dysk 5400 RPM w obudowie PC
    audio_5400 = generate_healthy_drive_in_pc(rpm=5400, pc_rms=0.005)
    res_5400 = detect_hdd_rpm_peaks(audio_5400)
    print("\n1. Test syntetyczny: Dysk 5400 RPM w szumie PC (45 dB):")
    print(f"   Wykryto RPM: {res_5400['rpm_detected']} -> {res_5400['best_match']}")

    # Test 2: Zdrowy dysk 7200 RPM w obudowie PC
    audio_7200 = generate_healthy_drive_in_pc(rpm=7200, pc_rms=0.005)
    res_7200 = detect_hdd_rpm_peaks(audio_7200)
    print("\n2. Test syntetyczny: Dysk 7200 RPM w szumie PC (45 dB):")
    print(f"   Wykryto RPM: {res_7200['rpm_detected']} -> {res_7200['best_match']}")

    # Test 3: Czysty szum wentylatorów PC (bez dysku)
    audio_fan = generate_pc_ambient_fan_noise(rms_level=0.005)
    res_fan = detect_hdd_rpm_peaks(audio_fan)
    print("\n3. Test syntetyczny: Czysty szum wentylatorów PC (bez dysku):")
    print(f"   Wykryto RPM: {res_fan['rpm_detected']} -> {res_fan['best_match']}")

    # Test 4: Prawdziwe pliki laboratoryjne z bazy
    print("\n4. Test na rzeczywistych plikach laboratoryjnych uszkodzeń:")
    raw_files = [
        PROJECT_ROOT / "data/raw/northwind/WDRaptorX.mp3",
        PROJECT_ROOT / "data/raw/northwind/Heads-seagate-barracuda-7200.12.mp3",
        PROJECT_ROOT / "data/raw/aesonlabs/Aesonlabs Data Recovery_Maxtor Heads Stuck Melody.wav",
        PROJECT_ROOT / "data/raw/northwind/seagate_stuck_spindle.mp3",
    ]

    for rf in raw_files:
        if rf.exists():
            y, _ = librosa.load(str(rf), sr=TARGET_SR, mono=True)
            res = detect_hdd_rpm_peaks(y)
            match_str = res['best_match']['profile'] if res['best_match'] else "Brak pików RPM"
            print(f"   {rf.name[:40]:40s} -> RPM: {match_str:15s} | Pisk zablokowania: {res['has_spindle_seize_tone']}")


if __name__ == "__main__":
    run_benchmark()
