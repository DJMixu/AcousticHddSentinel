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

# Profile prędkości obrotowych HDD z dynamicznym oknem śledzenia częstotliwości podstawowej
RPM_PROFILES = {
    "5400_rpm": {
        "nominal_rpm": 5400,
        "f0": 90.0,
        "f_range": (85.5, 94.5),  # +/- 4.5 Hz (tolerancja degradacji łożysk)
        "multipliers": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 12.0],
    },
    "7200_rpm": {
        "nominal_rpm": 7200,
        "f0": 120.0,
        "f_range": (114.0, 126.0),  # +/- 6.0 Hz (tolerancja degradacji łożysk)
        "multipliers": [2.0, 3.0, 4.0, 5.0, 6.0, 12.0],
    },
    "10000_rpm": {
        "nominal_rpm": 10000,
        "f0": 166.67,
        "f_range": (160.0, 173.0),
        "multipliers": [2.0, 3.0, 4.0, 6.0, 12.0],
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
    f_range: Tuple[float, float],
    multipliers: List[float],
    bandwidth_hz: float = 3.5
) -> Dict[str, Any]:
    """
    Dynamic Harmonic Tracking via Pitch Harmonizer:
    Wyszukuje optymalny f0 w przedziale f_range (uwzględniając spadek prędkości obrotowej wywołany uszkodzeniem łożysk),
    a następnie dynamicznie śledzi wyższe harmoniczne h_freq = m * f0_exact_freq.
    """
    total_energy = np.sum(spec**2) + 1e-12
    
    # 1. Sprawdzenie piku podstawowego f0 w dynamicznym oknie f_range
    f_min, f_max = f_range
    idx_f0 = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    if len(idx_f0) == 0:
        return {"detected": False, "score": 0.0, "f0_found": 0.0, "prominence": 0.0}
    
    f0_max_idx = idx_f0[np.argmax(spec[idx_f0])]
    f0_peak_val = spec[f0_max_idx]
    f0_exact_freq = freqs[f0_max_idx]
    
    # Tło lokalne wokół f0 (poza pasmem piku)
    local_bg_idx = np.where((freqs >= f0_target - 25.0) & (freqs <= f0_target + 25.0))[0]
    local_bg_idx = [i for i in local_bg_idx if abs(freqs[i] - f0_exact_freq) > bandwidth_hz]
    local_bg = np.median(spec[local_bg_idx]) if len(local_bg_idx) > 0 else 1e-6
    
    f0_prominence = f0_peak_val / (local_bg + 1e-9)
    
    # 2. Dynamiczne śledzenie wyższych harmonicznych proporcjonalnie do f0_exact_freq
    harmonic_hits = 0
    harmonic_energy = (f0_peak_val ** 2)
    has_2nd_harmonic_hit = False
    spec_mean = float(np.mean(spec))
    
    for m in multipliers:
        h_freq = f0_exact_freq * m
        if h_freq >= (TARGET_SR / 2.0) - 60.0:
            continue

        h_tol = max(bandwidth_hz * 1.5, h_freq * 0.025)
        idx_h = np.where((freqs >= h_freq - h_tol) & (freqs <= h_freq + h_tol))[0]
        if len(idx_h) > 0:
            h_peak = np.max(spec[idx_h])
            # Tło lokalne dla danej harmonicznej: dynamiczne skalowanie promienia okna
            # zapobiega wyzerowaniu liczby binów tła gdy h_tol przekracza sztywne 30 Hz (dla f > 1200 Hz)
            h_bg_window = max(40.0, h_tol * 2.0)
            h_bg_idx = np.where((freqs >= h_freq - h_bg_window) & (freqs <= h_freq + h_bg_window))[0]
            h_bg_idx = [i for i in h_bg_idx if abs(freqs[i] - h_freq) > h_tol]
            h_bg = np.median(spec[h_bg_idx]) if len(h_bg_idx) > 0 else 1e-6
            
            # Wymóg: wyrazistość względem lokalnego tła (> 1.6x) ORAZ bezwzględna energia względem średniej widma (> 5%)
            if (h_peak / (h_bg + 1e-9) > 1.6) and (h_peak > spec_mean * 0.05):
                harmonic_hits += 1
                harmonic_energy += (h_peak ** 2)
                if abs(m - 2.0) < 0.01:
                    has_2nd_harmonic_hit = True
                
    # Wyznaczenie ostrości piku f0 względem bocznych ramion (+/- 2 biny widma)
    if f0_max_idx >= 2 and f0_max_idx + 2 < len(spec):
        skirt = (spec[f0_max_idx - 2] + spec[f0_max_idx + 2]) / 2.0
        f0_sharpness = float(f0_peak_val / (skirt + 1e-9))
    else:
        f0_sharpness = 1.0

    comb_score = (f0_prominence * 0.4) + (harmonic_hits / len(multipliers) * 6.0)
    energy_ratio = float(harmonic_energy / total_energy)

    # Kryteria detekcji rotacji HDD:
    # 1. Multi-harmonic comb: harmonic_hits >= 2 i comb_score >= 3.0 oraz f0_prominence >= 2.2
    #    (lub f0_prominence >= 1.8 przy min. 3 harmonicznych i comb_score >= 3.8)
    # 2. Silny 2-tonowy rotor z f0 i 2f0 (np. autentyczne silniki FDB): harmonic_hits >= 1,
    #    f0_prominence >= 3.0, comb_score >= 3.0, POD WARUNKIEM że trafienie przypada na 2. harmoniczną (m=2.0)
    #    oraz ostrość piku f0_sharpness >= 6.0 (wyklucza szerokie wybrzuszenia modulacji wibrato głosu)
    multi_hit_comb = (
        harmonic_hits >= 2
        and comb_score >= 3.0
        and (
            f0_prominence >= 2.2
            or (f0_prominence >= 1.8 and harmonic_hits >= 3 and comb_score >= 3.8)
        )
    )
    single_2nd_harmonic_comb = (
        harmonic_hits >= 1
        and has_2nd_harmonic_hit
        and f0_prominence >= 3.0
        and comb_score >= 3.0
        and f0_sharpness >= 6.0
    )

    comb_detected = (
        (multi_hit_comb or single_2nd_harmonic_comb)
        and (energy_ratio >= 0.005 or harmonic_hits >= 4)
    )
    
    return {
        "detected": bool(comb_detected),
        "score": float(comb_score),
        "f0_found": float(f0_exact_freq),
        "f0_prominence": float(f0_prominence),
        "f0_sharpness": float(f0_sharpness),
        "harmonics_matched": int(harmonic_hits),
        "harmonics_total": len(multipliers),
        "harmonic_energy_ratio": float(harmonic_energy / total_energy)
    }


def verify_temporal_pitch_stationarity(
    y: np.ndarray,
    f0_found: float,
    sr: int = TARGET_SR,
    max_delta_hz: float = 1.5,
    n_fft: int = 4096
) -> bool:
    """
    Weryfikacja stacjonarności częstotliwości podstawowej f0 w czasie.
    Autentyczny silnik HDD wykazuje sztywną mechaniczną stabilność obrotów (wow & flutter < 0.05%, dryf < 0.2 Hz).
    Głos ludzki (wibrato) oraz syreny charakteryzują się dynamiczną modulacją częstotliwości (FM delta >= 3.0 Hz).
    Odrzuca sygnały, jeśli różnica częstotliwości f0 pomiędzy oknami przekracza max_delta_hz.
    """
    if len(y) < sr * 2.5:
        return True

    # 5 niespójnych (incommensurate) przesunięć czasowych w oknie 3.0s,
    # eliminujących ryzyko rezonansu fazowego ze standardowymi częstotliwościami wibrato mowy (3, 5, 8 Hz)
    offsets = [0, int(0.37 * sr), int(0.83 * sr), int(1.41 * sr), int(2.11 * sr)]
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    bin_width = sr / n_fft

    k_min = max(1, int(np.ceil((f0_found - 5.0) / bin_width)))
    k_max = min(n_fft // 2 - 1, int(np.floor((f0_found + 5.0) / bin_width)))
    if k_min > k_max:
        k_min = max(1, int(np.round((f0_found - 5.0) / bin_width)))
        k_max = min(n_fft // 2 - 1, int(np.round((f0_found + 5.0) / bin_width)))

    f_subs = []
    for off in offsets:
        if off + n_fft > len(y):
            continue
        chunk = y[off : off + n_fft] * hann
        spec = np.abs(np.fft.rfft(chunk, n=n_fft))

        k_peak = k_min + int(np.argmax(spec[k_min : k_max + 1]))
        alpha = float(spec[k_peak - 1])
        beta = float(spec[k_peak])
        gamma = float(spec[k_peak + 1])
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        delta = (gamma - alpha) / denom if denom != 0.0 else 0.0
        f_sub = (k_peak + delta) * bin_width
        f_subs.append(f_sub)

    if len(f_subs) < 3:
        return True

    delta_f = max(f_subs) - min(f_subs)
    return bool(delta_f <= max_delta_hz)


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
    
    # 1. Weryfikacja profili RPM z dynamicznym śledzeniem harmonicznych
    best_rpm = None
    best_rpm_data = None
    best_score = 0.0
    
    for rpm_key, profile in RPM_PROFILES.items():
        res = evaluate_harmonic_comb(freqs, spec, profile["f0"], profile["f_range"], profile["multipliers"])
        if res["detected"] and res["score"] > best_score:
            best_score = res["score"]
            best_rpm = rpm_key
            best_rpm_data = res

    # Weryfikacja stacjonarności rotacji wrzeciona (odrzucenie modulacji głosu i syren)
    if best_rpm is not None and best_rpm_data is not None:
        f0_detected = best_rpm_data["f0_found"]
        if not verify_temporal_pitch_stationarity(y, f0_detected, sr=sr):
            best_rpm = None
            best_rpm_data = None
            best_score = 0.0

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
    # Zatarte silniki wrzeciona (Seized Spindle) emitują ostry pisk między 1050 a 3500 Hz
    # przy braku prawidłowej, stabilnej rotacji silnika (has_rotation == False)!
    # Wymóg min. 0.5% energii w paśmie 1050 - 3500 Hz chroni przed cichymi szumami tła i cewkami zasilaczy.
    has_high_stall_beep = False
    stall_beep_freq = 0.0
    if not has_hdd_rotation:
        m_band = (freqs >= 1050.0) & (freqs <= 3500.0)
        beep_band_energy_ratio = float(np.sum(spec[m_band]**2) / (np.sum(spec**2) + 1e-12))
        if beep_band_energy_ratio >= 0.005:
            for p in top_peaks:
                f_p = p["frequency_hz"]
                if 1050.0 <= f_p <= 3500.0:
                    # Weryfikacja wyrazistości piku względem lokalnego tła w otoczeniu +/- 65 Hz
                    m_bg = (freqs >= f_p - 65.0) & (freqs <= f_p + 65.0) & (np.abs(freqs - f_p) > 12.0)
                    local_bg = float(np.median(spec[m_bg])) if np.sum(m_bg) > 0 else float(np.mean(spec))
                    prominence_ratio = p["amplitude"] / (local_bg + 1e-9)
                    if (p["prominence"] > np.mean(spec) * 1.5 or p["amplitude"] > np.mean(spec) * 2.0) and prominence_ratio >= 2.0:
                        has_high_stall_beep = True
                        stall_beep_freq = f_p
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
