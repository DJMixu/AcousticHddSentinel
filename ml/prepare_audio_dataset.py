"""
Generator i organizator zbioru danych audio dla Acoustic HDD Sentinel.
Single Source of Truth: ADR-001, fizyka akustyczna HDD & AGENTS.md

Główne zadania:
1. Pobiera autentyczne nagrania uszkodzeń z bazy data/raw/ (Northwind, Aesonlabs, Additional, Verified Real).
2. Automatycznie normalizuje i wzmacnia zbyt ciche pliki (np. WD5000AAKS, wd_laptop_clicking), aby RMS >= 0.002.
3. Wzbogaca klasę 'normal' o:
   - Autentyczne nagrania zdrowych dysków (Toshiba N300 7200 RPM, Seagate 7200 RPM)
   - Autentyczne nagrania szumu wentylatorów PC i tła komputera (~40-45 dB SPL)
   - Fizycznie symulowane dyski 5400 RPM (fundamentalna 90 Hz + harmoniczne)
   - Fizycznie symulowane dyski 7200 RPM (fundamentalna 120 Hz + harmoniczne)
   - Fizycznie symulowane wentylatory obudowy PC i chłodzenia CPU/GPU
4. Zapisuje zbiór do katalogu dataset_categorized/{clicking, beeping, scratching, normal}.
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

TARGET_SR = 22050
DATA_RAW_DIR = Path("data/raw")
RAW_NORTHWIND_DIR = DATA_RAW_DIR / "northwind"
RAW_ADDITIONAL_DIR = DATA_RAW_DIR / "additional"
RAW_AESON_DIR = DATA_RAW_DIR / "aesonlabs"
RAW_VERIFIED_DIR = DATA_RAW_DIR / "verified_real"
RAW_AMBIENT_DIR = DATA_RAW_DIR / "computer_ambient"
RAW_OUTLIERS_DIR = DATA_RAW_DIR / "outliers_oe"
CATEGORIZED_DIR = Path("dataset_categorized")


def check_and_boost_audio(y: np.ndarray, target_rms: float = 0.025) -> np.ndarray:
    """Wzmacnia zbyt ciche nagrania (RMS < 0.002), aby uniknąć odrzucenia przez bramkę DSP."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    if 0.0 < rms < 0.002:
        gain = target_rms / rms
        return (y * gain).astype(np.float32)
    return y.astype(np.float32)


def generate_healthy_hdd_sound(duration_sec: float = 14.0, rpm: int = 7200, sr: int = TARGET_SR, add_seeks: bool = True) -> np.ndarray:
    """
    Syntetyzuje fizycznie realistyczny dźwięk prawidłowo działającego dysku twardego HDD:
    - Buczenie obrotowe silnika wrzeciona (fundamentalna f0 = RPM / 60 + harmoniczne)
    - Szum aerodynamiczny powietrza i łożysk hydrodynamicznych (filtrowany szum)
    - Subtelne operacje pozycjonowania głowic (delikatne seek thuds)
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    f_base = (rpm * np.random.uniform(0.995, 1.005)) / 60.0
    flutter = 1.0 + 0.001 * np.sin(2 * np.pi * 0.25 * t)
    phase = 2 * np.pi * f_base * np.cumsum(flutter) / sr
    rand_phi = np.random.uniform(0, 2 * np.pi, size=6)

    # Harmoniczne obrotowe i biegunowe silnika
    motor_sound = 0.40 * np.sin(phase + rand_phi[0])
    motor_sound += 0.25 * np.sin(2 * phase + rand_phi[1])
    motor_sound += 0.15 * np.sin(3 * phase + rand_phi[2])
    motor_sound += 0.10 * np.sin(4 * phase + rand_phi[3])
    motor_sound += 0.06 * np.sin(6 * phase + rand_phi[4])
    motor_sound += 0.04 * np.sin(12 * phase + rand_phi[5])

    # Szum hydrodynamiczny łożysk FDB i oporu powietrza
    white_noise = np.random.randn(num_samples)
    upper_cutoff = 4500 if rpm <= 5400 else np.random.uniform(5500, 8000)
    sos_air = signal.butter(3, [100 / (sr / 2), upper_cutoff / (sr / 2)], btype="bandpass", output="sos")
    air_noise = signal.sosfilt(sos_air, white_noise) * (0.28 if rpm <= 5400 else 0.38)

    # Szum podłogowy mikrofonu (~45 dB SPL)
    ambient_noise = np.random.randn(num_samples) * 0.03
    sos_low = signal.butter(2, 600 / (sr / 2), btype="lowpass", output="sos")
    ambient_noise = signal.sosfilt(sos_low, ambient_noise)

    combined = motor_sound + air_noise + ambient_noise

    if add_seeks:
        num_seeks = np.random.randint(2, 6)
        seek_positions = np.random.uniform(1.0, duration_sec - 1.0, size=num_seeks)
        for s_pos in seek_positions:
            idx = int(s_pos * sr)
            seek_len = int(sr * 0.035)
            if idx + seek_len < num_samples:
                env = np.hanning(seek_len)
                seek_freq = np.random.uniform(400, 700)
                seek_tone = np.sin(2 * np.pi * seek_freq * np.linspace(0, 0.035, seek_len))
                combined[idx:idx + seek_len] += 0.25 * env * seek_tone

    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = (combined / peak) * 0.45

    return combined.astype(np.float32)


def generate_computer_fan_and_ambient(duration_sec: float = 14.0, fan_rpm: int = 1200, sr: int = TARGET_SR) -> np.ndarray:
    """
    Syntetyzuje fizycznie dokładny szum chłodzenia komputera (wentylatory obudowy, chłodzenie CPU/GPU, szum powietrza).
    - Brak ostrych wąskich pików rotacyjnych HDD (90 Hz / 120 Hz)
    - Szum wirowy (vortex shedding) i częstotliwość łopatek (Blade Pass Frequency = RPM/60 * blades)
    - Typowy szum komputera o głośności ~40-45 dB SPL
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    num_blades = np.random.choice([7, 9, 11])
    bpf = (fan_rpm / 60.0) * num_blades

    white = np.random.randn(num_samples)
    # Szum aerodynamiczny powietrza przepływającego przez żebra radiatora / grill obudowy
    sos_fan = signal.butter(3, [120 / (sr / 2), 3500 / (sr / 2)], btype="bandpass", output="sos")
    fan_noise = signal.sosfilt(sos_fan, white) * 0.40

    # Ton łopatek (BPF) z lekkim szumem turbulencji
    bpf_tone = 0.08 * np.sin(2 * np.pi * bpf * t)
    bpf_tone += 0.04 * np.sin(2 * np.pi * 2 * bpf * t)

    # Tło elektryczne zasilacza (przydźwięk sieciowy 50 Hz / 100 Hz o małej amplitudzie)
    hum_50hz = 0.02 * np.sin(2 * np.pi * 50.0 * t) + 0.01 * np.sin(2 * np.pi * 100.0 * t)

    combined = fan_noise + bpf_tone + hum_50hz
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = (combined / peak) * 0.35

    return combined.astype(np.float32)


def categorize_real_files(output_dir: Path = CATEGORIZED_DIR):
    """Kopiuje i kategoryzuje nagrania uszkodzeń oraz autentyczne pliki zdrowych dysków."""
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
    # 4. Zweryfikowane nagrania rzeczywiste
    if RAW_VERIFIED_DIR.exists():
        for vf in RAW_VERIFIED_DIR.glob("*.*"):
            if vf.suffix.lower() in [".wav", ".ogg", ".mp3"]:
                candidate_files.append(vf)

    print(f"Znaleziono {len(candidate_files)} plików źródłowych.")
    counts = {"scratching": 0, "beeping": 0, "clicking": 0, "normal": 0}

    specific_overrides = {
        "wd_250gb_stuck_spindle_chuttering.mp3": "beeping",
        "wd_500gb_bad_bearings.mp3": "beeping",
        "ibm_40gb_bad_heads_rattles_squeals.mp3": "scratching",
        "Aesonlabs Data Recovery_Seagate 8TB Clicking Due To Scratches ST8000DM004.wav": "clicking",
        # Nowe zweryfikowane nagrania autentyczne
        "TOSHIBA_HDWR480_7200rpm_healthy_spinup.wav": "normal",
        "Seagate_ST3250823A_7200rpm_healthy_seek.ogg": "normal",
        "WD102AA_Scratched_area_read_retries.ogg": "scratching",
        "WD_bad_heads_click_of_death.ogg": "clicking",
    }

    for p in tqdm(candidate_files, desc="Kategoryzacja audio"):
        fname = p.name
        nl = fname.lower()

        if fname in specific_overrides:
            cls = specific_overrides[fname]
        elif any(k in nl for k in ["spinup", "healthy", "spin_normal"]):
            cls = "normal"
        elif any(k in nl for k in ["surface", "scratch", "ring", "grinding", "drilling", "screech", "bad_media", "bad_sectors"]):
            cls = "scratching"
        elif any(k in nl for k in ["stiction", "spindle", "lock", "siren", "melody", "buzzing", "beep"]):
            cls = "beeping"
        else:
            cls = "clicking"

        dest = output_dir / cls / fname
        
        # Wczytanie i sprawdzenie czy plik nie wymaga podbicia RMS
        try:
            y, sr = librosa.load(str(p), sr=TARGET_SR)
            y_boosted = check_and_boost_audio(y)
            out_dest = dest.with_suffix(".wav")
            sf.write(str(out_dest), y_boosted, TARGET_SR, subtype="PCM_16")
            counts[cls] += 1
        except Exception as e:
            print(f"  [Błąd wczytywania] {p}: {e}")
            shutil.copy2(p, dest)
            counts[cls] += 1

    # 5. Dołączenie nagrań wentylatorów komputera do klasy 'normal'
    if RAW_AMBIENT_DIR.exists():
        ambient_files = list(RAW_AMBIENT_DIR.glob("*.wav")) + list(RAW_AMBIENT_DIR.glob("*.ogg"))
        for af in ambient_files:
            try:
                y, sr = librosa.load(str(af), sr=TARGET_SR)
                out_name = f"ambient_{af.stem}.wav"
                sf.write(str(output_dir / "normal" / out_name), y, TARGET_SR, subtype="PCM_16")
                counts["normal"] += 1
            except Exception as e:
                print(f"  [Błąd wczytywania tła] {af}: {e}")

    # 6. Dołączenie próbek Outlier Exposure (OE: usta, gwizd, mowa, klawiatura, szumy) do klasy 'normal'
    if RAW_OUTLIERS_DIR.exists():
        oe_files = list(RAW_OUTLIERS_DIR.glob("*.wav"))
        for oef in oe_files:
            try:
                y, sr = librosa.load(str(oef), sr=TARGET_SR)
                y_boosted = check_and_boost_audio(y)
                out_dest = output_dir / "normal" / oef.name
                sf.write(str(out_dest), y_boosted, TARGET_SR, subtype="PCM_16")
                counts["normal"] += 1
            except Exception as e:
                print(f"  [Błąd wczytywania OE] {oef}: {e}")

    print("\nPodsumowanie kategoryzacji zbioru danych:")
    for c, cnt in counts.items():
        print(f"  - {c:12s}: {cnt} plików")


def generate_comprehensive_normal_class(output_dir: Path = CATEGORIZED_DIR):
    """
    Generuje zróżnicowany, reprezentatywny zestaw próbek dla klasy 'normal':
    - 15 próbek zdrowych dysków 5400 RPM (fundamentalna 90 Hz)
    - 15 próbek zdrowych dysków 7200 RPM (fundamentalna 120 Hz)
    - 15 próbek szumu wentylatorów komputera / tła PC (~40-45 dB SPL)
    """
    normal_dir = output_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerowanie zrównoważonego podzbioru klasy 'normal' (5400 RPM, 7200 RPM, wentylatory PC)...")

    # 1. Zdrowe dyski 5400 RPM
    for i in range(1, 16):
        dur = np.random.uniform(12.0, 18.0)
        y = generate_healthy_hdd_sound(duration_sec=dur, rpm=5400, sr=TARGET_SR, add_seeks=(i % 2 == 0))
        out_name = f"synth_healthy_hdd_5400rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y, TARGET_SR, subtype="PCM_16")

    # 2. Zdrowe dyski 7200 RPM
    for i in range(1, 16):
        dur = np.random.uniform(12.0, 18.0)
        y = generate_healthy_hdd_sound(duration_sec=dur, rpm=7200, sr=TARGET_SR, add_seeks=(i % 2 == 0))
        out_name = f"synth_healthy_hdd_7200rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y, TARGET_SR, subtype="PCM_16")

    # 3. Szum wentylatorów komputera i tła obudowy PC (~45 dB SPL)
    for i in range(1, 16):
        dur = np.random.uniform(12.0, 18.0)
        fan_rpm = int(np.random.choice([900, 1200, 1500, 1800]))
        y = generate_computer_fan_and_ambient(duration_sec=dur, fan_rpm=fan_rpm, sr=TARGET_SR)
        out_name = f"synth_computer_fan_{fan_rpm}rpm_{i:02d}.wav"
        sf.write(str(normal_dir / out_name), y, TARGET_SR, subtype="PCM_16")

    total_normal = len(list(normal_dir.glob("*.wav")))
    print(f"Łącznie w klasie 'normal' znajduje się teraz {total_normal} plików audio.")


if __name__ == "__main__":
    categorize_real_files()
    generate_comprehensive_normal_class()
    print("\n[OK] Zbiór audio w dataset_categorized przygotowany pomyślnie!")
