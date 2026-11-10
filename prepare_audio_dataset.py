"""
Generator i organizator zbioru danych audio dla Acoustic HDD Sentinel.
1. Generuje realistyczne próbki audio prawidłowej pracy dysków (klasa 'normal').
2. Kategoryzuje istniejące nagrania uszkodzeń z CNN/dataset do klas: 'clicking', 'beeping', 'scratching'.
3. Zapisuje zbiór do katalogu CNN/dataset_categorized/{clicking, beeping, scratching, normal}.
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
SOURCE_DATASET_DIR = Path("CNN/dataset")
CATEGORIZED_DIR = Path("CNN/dataset_categorized")


def generate_healthy_hdd_sound(duration_sec: float = 12.0, rpm: int = 7200, sr: int = TARGET_SR, add_seeks: bool = True) -> np.ndarray:
    """
    Syntetyzuje fizycznie realistyczny dźwięk prawidłowo działającego dysku twardego HDD:
    - Buczenie obrotowe silnika wrzeciona (fundamentalna częstotliwość obrotowa + harmoniczne)
    - Szum aerodynamiczny powietrza i łożysk hydrodynamicznych (filtrowany szum różowy)
    - Subtelne, sporadyczne operacje pozycjonowania głowic (delikatne seek thuds)
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # 1. Częstotliwość podstawowa wrzeciona
    f_base = rpm / 60.0  # np. 120 Hz dla 7200 RPM, 90 Hz dla 5400 RPM
    
    # Drobne fluktuacje prędkości obrotowej (wow/flutter)
    flutter = 1.0 + 0.002 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * f_base * np.cumsum(flutter) / sr

    # Harmoniczne silnika i biegunów stojana
    motor_sound = 0.40 * np.sin(phase)                    # 1x f_base
    motor_sound += 0.25 * np.sin(2 * phase)                # 2x (druga harmoniczna)
    motor_sound += 0.15 * np.sin(3 * phase + 0.5)          # 3x
    motor_sound += 0.10 * np.sin(4 * phase + 1.2)          # 4x
    motor_sound += 0.06 * np.sin(6 * phase)                # 6x (bieguny stojana)
    motor_sound += 0.04 * np.sin(12 * phase)               # 12x

    # 2. Szum łożysk i oporu powietrza wirujących talerzy (szum różowy / filtrowany)
    white_noise = np.random.randn(num_samples)
    # Filtr pasmowo-przepustowy 150 Hz - 2400 Hz symulujący obudowę dysku
    sos_air = signal.butter(4, [150 / (sr / 2), 2400 / (sr / 2)], btype="bandpass", output="sos")
    air_noise = signal.sosfilt(sos_air, white_noise) * 0.35

    # 3. Delikatny szum tła otoczenia
    ambient_noise = np.random.randn(num_samples) * 0.05
    sos_low = signal.butter(2, 800 / (sr / 2), btype="lowpass", output="sos")
    ambient_noise = signal.sosfilt(sos_low, ambient_noise)

    combined = motor_sound + air_noise + ambient_noise

    # 4. Sporadyczne, miękkie ruchy pozycjonera głowic (soft head seeks)
    if add_seeks:
        num_seeks = np.random.randint(2, 6)
        seek_positions = np.random.uniform(1.0, duration_sec - 1.0, size=num_seeks)
        for s_pos in seek_positions:
            idx = int(s_pos * sr)
            seek_len = int(sr * 0.04)  # 40 ms
            if idx + seek_len < num_samples:
                # Miękki impuls gaussa (stłumiony dźwięk pozycjonowania w zdrowym dysku)
                env = np.hanning(seek_len)
                seek_freq = np.random.uniform(400, 750)
                seek_tone = np.sin(2 * np.pi * seek_freq * np.linspace(0, 0.04, seek_len))
                combined[idx:idx + seek_len] += 0.35 * env * seek_tone

    # Normalizacja amplitudy do realistycznego zakresu zdrowego dysku (RMS ~ 0.03 - 0.08)
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = (combined / peak) * 0.45

    return combined.astype(np.float32)


def categorize_existing_files():
    """
    Kategoryzuje pliki z CNN/dataset do klas:
    - clicking (Click of Death, stukanie głowicy)
    - beeping (zatarte wrzeciono, piski, melodie)
    - scratching (tarcie głowicy o talerz, zgrzytanie)
    """
    for cls in ["clicking", "beeping", "scratching", "normal"]:
        (CATEGORIZED_DIR / cls).mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(SOURCE_DATASET_DIR.glob("*.wav")))
    print(f"Znaleziono {len(wav_files)} istniejących plików audio w {SOURCE_DATASET_DIR}.")

    counts = {"clicking": 0, "beeping": 0, "scratching": 0}

    for fpath in tqdm(wav_files, desc="Kategoryzacja nagrań uszkodzeń"):
        fname_lower = fpath.name.lower()

        # 1. Reguły na podstawie jawnych nazw Aesonlabs
        if "beeping" in fname_lower or "melody" in fname_lower:
            target_class = "beeping"
        elif "scratch" in fname_lower or "screech" in fname_lower:
            target_class = "scratching"
        elif "clicking" in fname_lower:
            target_class = "clicking"
        else:
            # 2. Reguły na podstawie sygnatury akustycznej dla plików Northwind
            try:
                y, sr = librosa.load(str(fpath), sr=TARGET_SR, mono=True)
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                if len(y_trimmed) < sr * 0.5:
                    y_trimmed = y

                peak = np.max(np.abs(y_trimmed))
                rms = np.sqrt(np.mean(y_trimmed ** 2)) + 1e-9
                crest_factor = peak / rms

                # Płaskość widma (niska = tonalna/beeping, wysoka = szum/scratching)
                flatness = np.mean(librosa.feature.spectral_flatness(y=y_trimmed))
                # Rolloff częstotliwościowy 95%
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, roll_percent=0.95))
                # Onset rate (impulsowość kliknięć)
                onset_env = librosa.onset.onset_strength(y=y_trimmed, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                dur = len(y_trimmed) / sr
                onset_rate = len(onsets) / (dur + 1e-6)

                if flatness < 0.003:
                    # Bardzo czysty ton / pisk / syrena zablokowanego wrzeciona
                    target_class = "beeping"
                elif rolloff > 8500 and crest_factor < 20.0 and flatness > 0.05:
                    # Ciągły szerokopasmowy pisk/tarcie o wysokiej energii wysokotonowej
                    target_class = "scratching"
                elif crest_factor > 18.0 or onset_rate > 1.2:
                    # Wyraźne impulsowe stukanie głowicy o ogranicznik
                    target_class = "clicking"
                else:
                    # Pozostałe przypadki (przeważająca większość awarii HDD to uszkodzenia głowic / clicking)
                    target_class = "clicking"

            except Exception as e:
                target_class = "clicking"

        shutil.copy2(fpath, CATEGORIZED_DIR / target_class / fpath.name)
        counts[target_class] += 1

    print("\nPodsumowanie kategoryzacji nagrań uszkodzeń:")
    for cls, cnt in counts.items():
        print(f"  - {cls:12s}: {cnt} nagrań")


def generate_normal_class(count: int = 25):
    """
    Generuje zróżnicowany zbiór nagrań normalnej pracy dysku HDD.
    """
    print(f"\nGenerowanie {count} syntetycznych nagrań prawidłowej pracy dysku (klasa 'normal')...")
    normal_dir = CATEGORIZED_DIR / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        rpm = np.random.choice([5400, 7200])
        dur = np.random.uniform(10.0, 18.0)
        has_seeks = (i % 2 == 0)
        y_normal = generate_healthy_hdd_sound(duration_sec=dur, rpm=rpm, sr=TARGET_SR, add_seeks=has_seeks)
        
        out_name = f"synthetic_healthy_hdd_{rpm}rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y_normal, TARGET_SR, subtype="PCM_16")

    print(f"Wygenerowano {count} plików .wav w {normal_dir}.")


if __name__ == "__main__":
    categorize_existing_files()
    generate_normal_class(count=25)
    print("\nPrzygotowanie zbioru zakończone!")
