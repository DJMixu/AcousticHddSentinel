"""
Audytor jakości i integralności wszystkich próbek audio w projekcie Acoustic HDD Sentinel.
Weryfikuje:
1. Wszystkie pliki w data/raw/ (Northwind, Aesonlabs, Additional)
2. Wszystkie pliki w dataset_categorized/ (beeping, clicking, normal, scratching)
3. Parametry fizyczne: czas trwania, częstotliwość próbkowania, szczyt (Peak), RMS, SNR
4. Wykrywanie uszkodzonych nagrań (błędy nagłówków MP3, cisza poniżej progu RMS, przesterowania)
5. Analiza widmowa: piki częstotliwościowe i obecność składowych silnika HDD
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.hdd_audio_encoder import TARGET_SR

def audit_file(filepath: Path) -> dict:
    try:
        y, orig_sr = librosa.load(str(filepath), sr=None, mono=True)
    except Exception as e:
        return {
            "file": filepath.name,
            "path": str(filepath),
            "status": "ERROR_CORRUPT",
            "error": str(e)
        }

    dur = len(y) / orig_sr
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    
    # Obliczenie spektrum FFT dla pierwszych 3 sekund
    sample_len = min(len(y), int(3.0 * orig_sr))
    fft = np.abs(np.fft.rfft(y[:sample_len]))
    freqs = np.fft.rfftfreq(sample_len, 1.0 / orig_sr)
    
    # Wyszukanie dominującego piku w zakresie 50 - 5000 Hz
    valid_idx = np.where((freqs >= 50.0) & (freqs <= 5000.0))[0]
    if len(valid_idx) > 0:
        top_idx = valid_idx[np.argmax(fft[valid_idx])]
        top_freq = float(freqs[top_idx])
    else:
        top_freq = 0.0

    # Sprawdzenie obecności energii w okolicach 90 Hz (5400 RPM) i 120 Hz (7200 RPM)
    idx_90 = np.argmin(np.abs(freqs - 90.0))
    idx_120 = np.argmin(np.abs(freqs - 120.0))
    energy_90 = float(fft[idx_90])
    energy_120 = float(fft[idx_120])
    mean_energy = float(np.mean(fft)) + 1e-9

    has_90hz_peak = (energy_90 / mean_energy) > 2.5
    has_120hz_peak = (energy_120 / mean_energy) > 2.5

    status = "OK"
    warnings = []
    if dur < 3.0:
        status = "WARNING_TOO_SHORT"
        warnings.append(f"Czas nagrania ({dur:.2f}s) krótszy niż 3.0s")
    if rms < 0.002:
        status = "WARNING_SILENCE"
        warnings.append(f"Zbyt niski poziom RMS ({rms:.5f} < 0.002) - odrzucany przez bramkę DSP")
    if peak > 0.999:
        warnings.append("Możliwy clipping / przesterowanie (Peak >= 1.0)")

    return {
        "file": filepath.name,
        "path": str(filepath),
        "status": status,
        "warnings": warnings,
        "duration_sec": round(dur, 2),
        "orig_sr": int(orig_sr),
        "rms": round(rms, 5),
        "peak": round(peak, 4),
        "top_freq_hz": round(top_freq, 1),
        "peak_90hz_prominence": round(energy_90 / mean_energy, 2),
        "peak_120hz_prominence": round(energy_120 / mean_energy, 2),
        "has_5400rpm_signature": has_90hz_peak,
        "has_7200rpm_signature": has_120hz_peak
    }

def run_full_audit():
    print("=" * 80)
    print("AUDYT WSZYSTKICH PRÓBEK AUDIO W PROJEKCIE (Ground Truth & Syntetyczne)")
    print("=" * 80)

    target_dirs = {
        "raw_northwind": PROJECT_ROOT / "data" / "raw" / "northwind",
        "raw_aesonlabs": PROJECT_ROOT / "data" / "raw" / "aesonlabs",
        "raw_additional": PROJECT_ROOT / "data" / "raw" / "additional",
        "categorized_beeping": PROJECT_ROOT / "dataset_categorized" / "beeping",
        "categorized_clicking": PROJECT_ROOT / "dataset_categorized" / "clicking",
        "categorized_normal": PROJECT_ROOT / "dataset_categorized" / "normal",
        "categorized_scratching": PROJECT_ROOT / "dataset_categorized" / "scratching",
    }

    full_report = {}
    total_audited = 0
    total_issues = 0

    for cat_name, dir_path in target_dirs.items():
        if not dir_path.exists():
            continue
        files = sorted(list(dir_path.glob("*.wav")) + list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.flac")))
        print(f"\nAnaliza katalogu: {cat_name} ({len(files)} plików)...")
        cat_results = []
        for f in files:
            res = audit_file(f)
            cat_results.append(res)
            total_audited += 1
            if res["status"] != "OK" or len(res["warnings"]) > 0:
                total_issues += 1
                warn_str = ", ".join(res["warnings"]) if res["warnings"] else res["status"]
                print(f"  [UWAGA] {f.name:50s} | RMS: {res.get('rms', 0):.5f} | Status: {warn_str}")
        full_report[cat_name] = cat_results

    out_json = PROJECT_ROOT / "tests" / "audit_samples_report.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(full_report, fp, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"PODSUMOWANIE AUDYTU: Przebadano {total_audited} plików, wykryto {total_issues} potencjalnych uwag/ostrzeżeń.")
    print(f"Pełny raport zapisano do: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    run_full_audit()
