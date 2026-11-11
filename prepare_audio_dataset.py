"""
Generator i organizator zbioru danych audio dla Acoustic HDD Sentinel.
Single Source of Truth: ADR-001 & AGENTS.md

1. Pobiera autentyczne nagrania uszkodzeń z bazy data/raw/ (Northwind, Aesonlabs, materiały laboratoryjne).
2. Przypisuje etykiety na podstawie laboratoryjnego Ground Truth (Northwind Data Recovery / Datacent / Aesonlabs).
3. Generuje 30 realistycznych, fizycznie syntezowanych próbek prawidłowej pracy dysków (klasa 'normal').
4. Zapisuje zbiór do katalogu dataset_categorized/{clicking, beeping, scratching, normal}.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
from tqdm import tqdm

TARGET_SR = 22050
DATA_RAW_DIR = Path("data/raw")
RAW_NORTHWIND_DIR = DATA_RAW_DIR / "northwind"
RAW_ADDITIONAL_DIR = DATA_RAW_DIR / "additional"
RAW_AESON_DIR = DATA_RAW_DIR / "aesonlabs"
CATEGORIZED_DIR = Path("dataset_categorized")


def generate_healthy_hdd_sound(duration_sec: float = 14.0, rpm: int = 7200, sr: int = TARGET_SR, add_seeks: bool = True) -> np.ndarray:
    """
    Syntetyzuje fizycznie realistyczny dźwięk prawidłowo działającego dysku twardego HDD:
    - Buczenie obrotowe silnika wrzeciona (fundamentalna częstotliwość obrotowa + harmoniczne)
    - Szum aerodynamiczny powietrza i łożysk hydrodynamicznych (filtrowany szum różowy)
    - Subtelne, sporadyczne operacje pozycjonowania głowic (delikatne seek thuds)
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    f_base = (rpm * np.random.uniform(0.985, 1.015)) / 60.0
    flutter = 1.0 + 0.002 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * f_base * np.cumsum(flutter) / sr
    rand_phi = np.random.uniform(0, 2 * np.pi, size=6)

    motor_sound = 0.40 * np.sin(phase + rand_phi[0])
    motor_sound += 0.25 * np.sin(2 * phase + rand_phi[1])
    motor_sound += 0.15 * np.sin(3 * phase + rand_phi[2])
    motor_sound += 0.10 * np.sin(4 * phase + rand_phi[3])
    motor_sound += 0.06 * np.sin(6 * phase + rand_phi[4])
    motor_sound += 0.04 * np.sin(12 * phase + rand_phi[5])

    white_noise = np.random.randn(num_samples)
    upper_cutoff = np.random.uniform(3500, 7500)
    sos_air = signal.butter(3, [120 / (sr / 2), upper_cutoff / (sr / 2)], btype="bandpass", output="sos")
    air_noise = signal.sosfilt(sos_air, white_noise) * 0.35

    ambient_noise = np.random.randn(num_samples) * 0.05
    sos_low = signal.butter(2, 800 / (sr / 2), btype="lowpass", output="sos")
    mic_floor = np.random.randn(num_samples) * np.random.uniform(0.008, 0.025)
    ambient_noise = signal.sosfilt(sos_low, ambient_noise) + mic_floor

    combined = motor_sound + air_noise + ambient_noise

    if add_seeks:
        num_seeks = np.random.randint(2, 6)
        seek_positions = np.random.uniform(1.0, duration_sec - 1.0, size=num_seeks)
        for s_pos in seek_positions:
            idx = int(s_pos * sr)
            seek_len = int(sr * 0.04)
            if idx + seek_len < num_samples:
                env = np.hanning(seek_len)
                seek_freq = np.random.uniform(400, 750)
                seek_tone = np.sin(2 * np.pi * seek_freq * np.linspace(0, 0.04, seek_len))
                combined[idx:idx + seek_len] += 0.35 * env * seek_tone

    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = (combined / peak) * 0.45

    return combined.astype(np.float32)


def categorize_real_files(output_dir: Path = CATEGORIZED_DIR):
    """Kopiuje i kategoryzuje nagrania uszkodzeń z dokładnym Ground Truth."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for c in ["clicking", "beeping", "scratching", "normal"]:
        (output_dir / c).mkdir(parents=True, exist_ok=True)

    candidate_files = []
    # 1. Northwind
    if RAW_NORTHWIND_DIR.exists():
        candidate_files.extend(list(RAW_NORTHWIND_DIR.glob("*.mp3")))
    # 2. Aesonlabs
    if RAW_AESON_DIR.exists():
        candidate_files.extend(list(RAW_AESON_DIR.glob("Aesonlabs*.wav")))
    # 3. Dodatkowe nagrania laboratoryjne
    if RAW_ADDITIONAL_DIR.exists():
        candidate_files.extend(list(RAW_ADDITIONAL_DIR.glob("*.mp3")))

    print(f"Znaleziono {len(candidate_files)} plików ze źródłami awarii.")

    counts = {"scratching": 0, "beeping": 0, "clicking": 0}

    # Jawne mapowanie dla plików o specyficznej charakterystyce
    specific_overrides = {
        "wd_250gb_stuck_spindle_chuttering.mp3": "beeping",
        "wd_500gb_bad_bearings.mp3": "beeping",
        "ibm_40gb_bad_heads_rattles_squeals.mp3": "scratching",
        # Plik Aesonlabs 8TB zawiera dominujące periodyczne klikanie głowic przy próbie odczytu porysowanego talerza
        "Aesonlabs Data Recovery_Seagate 8TB Clicking Due To Scratches ST8000DM004.wav": "clicking",
    }

    for p in tqdm(candidate_files, desc="Kategoryzacja audio"):
        fname = p.name
        nl = fname.lower()

        if fname in specific_overrides:
            cls = specific_overrides[fname]
        elif any(k in nl for k in ["surface", "scratch", "ring", "grinding", "drilling", "screech", "bad_media", "bad_sectors"]):
            cls = "scratching"
        elif any(k in nl for k in ["stiction", "spindle", "lock", "siren", "melody", "buzzing", "beep"]):
            cls = "beeping"
        else:
            cls = "clicking"

        dest = output_dir / cls / fname
        shutil.copy2(p, dest)
        counts[cls] += 1

    print("\nPodsumowanie kategoryzacji uszkodzeń:")
    for c, cnt in counts.items():
        print(f"  - {c:12s}: {cnt} plików")


def generate_normal_class(count: int = 30, output_dir: Path = CATEGORIZED_DIR):
    """Generuje realistyczne próbki audio prawidłowej pracy dysków (klasa 'normal')."""
    print(f"\nGenerowanie {count} nagrań prawidłowej pracy dysku (klasa 'normal')...")
    normal_dir = output_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        rpm = np.random.choice([5400, 7200])
        dur = np.random.uniform(11.0, 18.0)
        has_seeks = (i % 2 == 0)
        y_normal = generate_healthy_hdd_sound(duration_sec=dur, rpm=rpm, sr=TARGET_SR, add_seeks=has_seeks)
        out_name = f"synthetic_healthy_hdd_{rpm}rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y_normal, TARGET_SR, subtype="PCM_16")

    print(f"Wygenerowano {count} plików w {normal_dir}.")


if __name__ == "__main__":
    categorize_real_files()
    generate_normal_class(count=30)
    print("\n[OK] Zbiór audio w dataset_categorized przygotowany pomyślnie!")
