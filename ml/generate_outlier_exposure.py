"""
Generowanie zbioru Outlier Exposure (OE) dla klasy 'normal'.
Single Source of Truth: ADR-001 & AGENTS.md

Skrypt generuje syntetyczne i półsyntetyczne próbki dźwięków obcych (Out-Of-Distribution),
które człowiek lub otoczenie komputera może wytworzyć:
1. Dźwięki ustami (klikanie językiem, cmokanie, gwizdanie, mowa, samogłoski)
2. Szumy wentylatorów PC i przepływu powietrza (różne RPM i turbulencje)
3. Odgłosy biurowe (klawiatura mechaniczna, kliknięcia myszką, stukanie w biurko)

Wszystkie te próbki są dodawane do klasy 'normal', dzięki czemu sieć neuronowa
aktywnie uczy się odróżniać mechaniczne uszkodzenia HDD od dźwięków otoczenia i człowieka.
"""

import os
import math
import numpy as np
import soundfile as sf
from typing import List, Tuple

TARGET_SR = 22050
DURATION = 3.5  # sekundy (> 3.0 s dla wycinków ADR-001)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "outliers_oe")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_mouth_clicks(duration: float = DURATION, sr: int = TARGET_SR, num_clicks: int = 5) -> np.ndarray:
    """Imitacja klikania językiem / ustami (formanty jamy ustnej 800-2000 Hz, miękkie tłumienie)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    click_times = np.linspace(0.4, duration - 0.4, num_clicks) + np.random.uniform(-0.1, 0.1, num_clicks)
    
    for ct in click_times:
        idx = int(ct * sr)
        click_len = int(0.04 * sr)  # ~40 ms
        if idx + click_len < len(sig):
            decay = np.exp(-np.linspace(0, 10, click_len))
            # Rezonans jamy ustnej (formanty ~900 Hz i ~1800 Hz)
            t_c = np.linspace(0, 0.04, click_len, endpoint=False)
            f1 = np.random.uniform(700, 1100)
            f2 = np.random.uniform(1600, 2200)
            res = (0.7 * np.sin(2 * np.pi * f1 * t_c) + 0.3 * np.sin(2 * np.pi * f2 * t_c)) * decay
            sig[idx:idx + click_len] += res
            
    # Lekki szum powietrza w tle
    sig += 0.005 * np.random.normal(0, 1, len(sig))
    return sig


def generate_whistle_and_voice(duration: float = DURATION, sr: int = TARGET_SR) -> np.ndarray:
    """Imitacja gwizdania i głosu ludzkiego (zmienna modulacja tonu podstawowego F0)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Zmieniająca się częstotliwość (modulacja gwizdu lub mowy ~1200-2400 Hz)
    f_center = np.random.uniform(1000, 2500)
    f_mod = f_center + 300 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f_mod) / sr
    sig = 0.4 * np.sin(phase)
    # 2. harmoniczna
    sig += 0.1 * np.sin(2 * phase)
    # Szum oddechu
    sig += 0.01 * np.random.normal(0, 1, len(sig))
    return sig


def generate_pc_fan_turbulent(duration: float = DURATION, sr: int = TARGET_SR) -> np.ndarray:
    """Szum wentylatora komputerowego z turbulencją wirową (różne RPM)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rpm = np.random.uniform(800, 2400)
    blade_pass = (rpm / 60.0) * np.random.choice([7, 9, 11])  # 7-11 łopatek
    
    # Szum różowy / turbulencja wirowa
    white = np.random.normal(0, 1, len(t))
    # Prosty filtr dolnoprzepustowy (integrator)
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    if np.max(np.abs(pink)) > 0:
        pink = pink / np.max(np.abs(pink))
        
    # Składowa harmoniczna łopatek wentylatora
    blade_tone = 0.15 * np.sin(2 * np.pi * blade_pass * t)
    blade_tone += 0.05 * np.sin(2 * np.pi * (2 * blade_pass) * t)
    
    sig = 0.3 * pink + blade_tone
    return sig


def generate_keyboard_typing(duration: float = DURATION, sr: int = TARGET_SR, num_presses: int = 6) -> np.ndarray:
    """Stukanie w klawiaturę mechaniczną / klikanie myszką."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    press_times = np.linspace(0.3, duration - 0.3, num_presses) + np.random.uniform(-0.15, 0.15, num_presses)
    
    for pt in press_times:
        idx = int(pt * sr)
        key_len = int(0.025 * sr)  # ~25 ms
        if idx + key_len < len(sig):
            decay = np.exp(-np.linspace(0, 14, key_len))
            noise = np.random.normal(0, 1, key_len) * decay
            sig[idx:idx + key_len] += noise
            
    sig += 0.003 * np.random.normal(0, 1, len(sig))
    return sig


def generate_room_ambient_hum(duration: float = DURATION, sr: int = TARGET_SR) -> np.ndarray:
    """Szum tła pokoju / klimatyzacji z przydźwiękiem sieciowym 50 Hz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    hum = 0.2 * np.sin(2 * np.pi * 50.0 * t) + 0.08 * np.sin(2 * np.pi * 100.0 * t)
    air = 0.15 * np.random.normal(0, 1, len(t))
    sig = hum + air
    return sig


def main():
    print(f"Generowanie próbek Outlier Exposure (OE) w {OUT_DIR}...")
    np.random.seed(42)
    
    generators = [
        ("oe_mouth_clicks", generate_mouth_clicks, 15),
        ("oe_whistle_voice", generate_whistle_and_voice, 12),
        ("oe_pc_fan_turbulent", generate_pc_fan_turbulent, 15),
        ("oe_keyboard_typing", generate_keyboard_typing, 10),
        ("oe_room_ambient_hum", generate_room_ambient_hum, 10),
    ]
    
    total_generated = 0
    for prefix, gen_func, count in generators:
        for i in range(count):
            samples = gen_func()
            # Normalizacja do bezpiecznego zakresu [-0.85, 0.85]
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = (samples / max_val) * 0.85
                
            fname = f"{prefix}_{i+1:02d}.wav"
            fpath = os.path.join(OUT_DIR, fname)
            sf.write(fpath, samples.astype(np.float32), TARGET_SR)
            total_generated += 1
            
    print(f"Sukces: Wygenerowano {total_generated} plików Outlier Exposure (OE).")


if __name__ == "__main__":
    main()
