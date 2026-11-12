"""
Moduł generowania próbek adwersarialnych do testów i odporności OOD.
Single Source of Truth: ADR-001, AGENTS.md oraz rekomendacje Qwen 3.8 / Qwen Coder.

Generuje trudne przypadki brzegowe:
1. GPU Coil Whine: wąskopasmowe harmoniczne PWM 1.5-8 kHz z modulacją obciążenia i brakiem mikrotransjentów.
2. Syntetyczny metronom: idealnie periodyczne impulsy (CV < 0.05) o stałej amplitudzie (120 BPM).
3. Ciągły pisk rezonansowy: czysty ton ze świstem harmonicznym.
"""

import numpy as np
from scipy.signal import sawtooth, square
from typing import Tuple

TARGET_SR = 22050
DURATION = 3.5  # sekundy (> 3.0 s dla wycinków ADR-001)


def generate_gpu_coil_whine(duration: float = DURATION, sr: int = TARGET_SR) -> np.ndarray:
    """
    Generuje realistyczny sygnał GPU Coil Whine z wąskopasmowymi harmonicznymi PWM (1.5 - 8 kHz).
    Zawiera modulację obciążenia GPU (duty-cycle) i brak mikrotransjentów tarcia mechanicznego.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Podstawowa częstotliwość przełączania przetwornicy VRM/PWM (np. 2.4 kHz, 3.8 kHz lub 5.2 kHz)
    f_base = np.random.uniform(2000, 3500)
    
    # Modulacja obciążenia GPU (np. pętla renderowania klatek 60-144 FPS)
    load_hz = np.random.uniform(30, 140)
    load_mod = 0.5 + 0.5 * np.sin(2 * np.pi * load_hz * t)
    
    # Harmoniczne magnetostrykcji cewek
    sig = np.zeros_like(t)
    harmonics = [1.0, 2.0, 3.0, 0.5]
    weights = [0.6, 0.25, 0.1, 0.15]
    
    for h, w in zip(harmonics, weights):
        f = f_base * h
        if f < sr / 2 - 500:
            # Wąskopasmowy ton z mikroskopijnym jitterem fazy
            phase = 2 * np.pi * f * t + 0.05 * np.sin(2 * np.pi * 3.0 * t)
            sig += w * np.sin(phase) * (0.8 + 0.2 * load_mod)
            
    # Subharmoniczna PWM (pisk zasilacza impulsowego)
    sig += 0.15 * sawtooth(2 * np.pi * (f_base * 0.5) * t, width=0.5)
    
    # Dodatek cichego szumu tła (elektroniczny szum termiczny)
    noise = np.random.normal(0, 0.005, len(t))
    sig += noise
    
    # Normalizacja szczytowa
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.6
        
    return sig.astype(np.float32)


def generate_synthetic_metronome(duration: float = DURATION, sr: int = TARGET_SR, bpm: int = 120) -> np.ndarray:
    """
    Generuje syntetyczny metronom z idealnie równomiernymi impulsami (niski jitter, stałe odstępy).
    Odróżnia się od Click of Death zerową wariancją interwałów (CV < 0.05).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    
    interval = 60.0 / bpm  # np. 0.5 s dla 120 BPM
    impulse_times = np.arange(0.2, duration - 0.1, interval)
    
    impulse_len = int(0.012 * sr)  # 12 ms klik
    t_imp = np.linspace(0, 0.012, impulse_len, endpoint=False)
    decay = np.exp(-t_imp * 350.0)
    # Rezonans kowadełka metronomu ~2200 Hz
    click_kernel = np.sin(2 * np.pi * 2200.0 * t_imp) * decay
    
    for it in impulse_times:
        idx = int(it * sr)
        if idx + impulse_len < len(sig):
            sig[idx:idx + impulse_len] += click_kernel
            
    # Bardzo niski szum tła
    sig += np.random.normal(0, 0.002, len(t))
    
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.7
        
    return sig.astype(np.float32)
