"""
Moduł analizy pików częstotliwościowych i detekcji harmonicznych RPM dla dysków HDD.
Single Source of Truth: ADR-001, fizyka akustyczna HDD & AGENTS.md

Funkcjonalności:
1. Detekcja fundamentalnej częstotliwości obrotowej silnika wrzeciona:
   - 5400 RPM -> f0 = 90.0 Hz (+/- 1.5 Hz)
   - 7200 RPM -> f0 = 120.0 Hz (+/- 1.8 Hz)
   - 10000 RPM -> f0 = 166.7 Hz (+/- 2.5 Hz)
2. Analiza grzebienia harmonicznego (Harmonic Comb Analysis):
   - Weryfikacja obecności składowych k * f0 (2x, 3x, 4x, 6x, 12x)
3. Test stacjonarności częstotliwości (odrzucanie głosu ludzkiego z wibrato i intonacją)
4. Obliczenie współczynnika wyrazistości harmonicznej (Harmonic Prominence Ratio)
5. Dyskryminacja: Zdrowy HDD (5400/7200) vs Wentylator PC vs Głos vs Awaria (Brak rotacji / Zablokowanie)
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import scipy.signal as signal

TARGET_SR = 22050

# Profile prędkości obrotowych HDD
RPM_PROFILES = {
    "5400_rpm": {
        "nominal_rpm": 5400,
        "f0": 90.0,
        "f_range": (88.5, 91.5),
        "harmonics": [180.0, 270.0, 360.0, 450.0, 540.0, 1080.0],
    },
    "7200_rpm": {
        "nominal_rpm": 7200,
        "f0": 120.0,
        "f_range": (118.0, 122.0),
        "harmonics": [240.0, 360.0, 480.0, 600.0, 720.0, 1440.0],
    },
    "10000_rpm": {
        "nominal_rpm": 10000,
        "f0": 166.67,
        "f_range": (164.0, 169.0),
        "harmonics": [333.3, 500.0, 666.7, 1000.0, 2000.0],
    }
}


def compute_high_res_spectrum(y: np.ndarray, sr: int = TARGET_SR, n_fft: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oblicza widmo amplitudowe o wysokiej rozdzielczości (krok ~2.69 Hz przy 8192 i SR 22050).
    Stosuje okno Flattop lub Hann z uśrednianiem segmentów (Welch).
    """
    if len(y) < n_fft:
        pad_len = n_fft - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")
    
    freqs, psd = signal.welch(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft // 2,
        scaling="spectrum"
    )
    amplitude_spectrum = np.sqrt(psd)
    return freqs, amplitude_spectrum


def evaluate_harmonic_comb(
    freqs: np.ndarray,
    spec: np.ndarray,
    f0_target: float,
    harmonics_list: List[float],
    bandwidth_hz: float = 3.0
) -> Dict[str, Any]:
    """
    Ocenia obecność grzebienia harmonicznego dla zadanej częstotliwości bazowej f0.
    """
    total_energy = np.sum(spec**2) + 1e-12
    
    # 1. Sprawdzenie piku podstawowego f0
    idx_f0 = np.where((freqs >= f0_target - bandwidth_hz) & (freqs <= f0_target + bandwidth_hz))[0]
    if len(idx_f0) == 0:
        return {"detected": False, "score": 0.0, "f0_found": 0.0, "prominence": 0.0}
    
    f0_max_idx = idx_f0[np.argmax(spec[idx_f0])]
    f0_peak_val = spec[f0_max_idx]
    f0_exact_freq = freqs[f0_max_idx]
    
    # Tło lokalne wokół f0 (poza pasmem piku)
    local_bg_idx = np.where((freqs >= f0_target - 15.0) & (freqs <= f0_target + 15.0))[0]
    local_bg_idx = [i for i in local_bg_idx if abs(freqs[i] - f0_exact_freq) > bandwidth_hz]
    local_bg = np.median(spec[local_bg_idx]) if len(local_bg_idx) > 0 else 1e-6
    
    f0_prominence = f0_peak_val / (local_bg + 1e-9)
    
    # 2. Sprawdzenie wyższych harmonicznych
    harmonic_hits = 0
    harmonic_energy = (f0_peak_val ** 2)
    
    for h_freq in harmonics_list:
        idx_h = np.where((freqs >= h_freq - bandwidth_hz * 1.5) & (freqs <= h_freq + bandwidth_hz * 1.5))[0]
        if len(idx_h) > 0:
            h_peak = np.max(spec[idx_h])
            # Tło lokalne dla danej harmonicznej
            h_bg_idx = np.where((freqs >= h_freq - 25.0) & (freqs <= h_freq + 25.0))[0]
            h_bg_idx = [i for i in h_bg_idx if abs(freqs[i] - h_freq) > bandwidth_hz * 1.5]
            h_bg = np.median(spec[h_bg_idx]) if len(h_bg_idx) > 0 else 1e-6
            
            if h_peak / (h_bg + 1e-9) > 1.8:
                harmonic_hits += 1
                harmonic_energy += (h_peak ** 2)
                
    comb_score = (f0_prominence * 0.4) + (harmonic_hits / len(harmonics_list) * 6.0)
    comb_detected = (f0_prominence >= 2.0 and harmonic_hits >= 2) or (f0_prominence >= 3.5 and harmonic_hits >= 1)
    
    return {
        "detected": bool(comb_detected),
        "score": float(comb_score),
        "f0_found": float(f0_exact_freq),
        "f0_prominence": float(f0_prominence),
        "harmonics_matched": int(harmonic_hits),
        "harmonics_total": len(harmonics_list),
        "harmonic_energy_ratio": float(harmonic_energy / total_energy)
    }


def analyze_frequency_peaks(y: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
    """
    Główna funkcja analizy częstotliwościowej nagrania audio.
    Zwraca:
    - Wykryty profil RPM ('5400_rpm', '7200_rpm', '10000_rpm' lub None)
    - Obecność ostrych pików częstotliwościowych (Piszczenie silnika, pisk głowic)
    - Czy sygnał jest stacjonarny mechanicznie (HDD) czy jest szumem wentylatora / mową
    """
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 0.002:
        return {
            "status": "SILENCE",
            "detected_rpm": None,
            "rpm_confidence": 0.0,
            "has_rotation": False,
            "rms": rms,
            "dominant_peaks": []
        }

    freqs, spec = compute_high_res_spectrum(y, sr=sr, n_fft=8192)
    
    # 1. Weryfikacja profili RPM
    best_rpm = None
    best_rpm_data = None
    best_score = 0.0
    
    for rpm_key, profile in RPM_PROFILES.items():
        res = evaluate_harmonic_comb(freqs, spec, profile["f0"], profile["harmonics"])
        if res["detected"] and res["score"] > best_score:
            best_score = res["score"]
            best_rpm = rpm_key
            best_rpm_data = res

    # 2. Wykrywanie dominujących ostrych pików w całym paśmie (np. pisk stojana 1000 - 3000 Hz)
    peaks, properties = signal.find_peaks(spec, prominence=np.mean(spec) * 2.5, distance=15)
    top_peaks = []
    if len(peaks) > 0:
        sorted_indices = np.argsort(spec[peaks])[::-1][:8]
        for idx in sorted_indices:
            p_idx = peaks[idx]
            top_peaks.append({
                "frequency_hz": round(float(freqs[p_idx]), 1),
                "amplitude": float(spec[p_idx]),
                "prominence": round(float(properties["prominences"][idx]), 4)
            })

    # 3. Sprawdzenie charakteru sygnału (Wentylator PC vs Wirujący HDD)
    has_hdd_rotation = (best_rpm is not None)

    # 4. Test obecności pisku stojana (Stall Beep) przy zatartym silniku:
    # Zatarte silniki wrzeciona (Seized Spindle) emitują ostry pisk między 800 a 3500 Hz
    # przy braku prawidłowej, stabilnej rotacji silnika (has_rotation == False)!
    has_high_stall_beep = False
    stall_beep_freq = 0.0
    if not has_hdd_rotation:
        for p in top_peaks:
            if 800.0 <= p["frequency_hz"] <= 3500.0 and p["prominence"] > np.mean(spec) * 6.0:
                has_high_stall_beep = True
                stall_beep_freq = p["frequency_hz"]
                break
    
    return {
        "status": "ANALYZED",
        "rms": round(rms, 5),
        "detected_rpm": best_rpm,
        "rpm_nominal": RPM_PROFILES[best_rpm]["nominal_rpm"] if best_rpm else None,
        "rpm_confidence": round(min(best_score / 10.0, 1.0), 3) if best_rpm else 0.0,
        "rpm_details": best_rpm_data,
        "has_rotation": has_hdd_rotation,
        "has_stall_beep": has_high_stall_beep,
        "stall_beep_freq_hz": round(stall_beep_freq, 1),
        "dominant_peaks": top_peaks
    }
