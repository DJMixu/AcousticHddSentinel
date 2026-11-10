import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# === KONFIGURACJA ===
DATASET_DIR = "./dataset"  # Wpisz ścieżkę do swojego głównego folderu z plikami audio
OUTPUT_CSV = "audio_dataset_profile.csv"
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

def analyze_audio_file(filepath: Path, root_dir: Path) -> dict:
    # Nazwa podfolderu jako etykieta klasy (lub 'root', jeśli pliki są bezpośrednio w głównym katalogu)
    rel_path = filepath.relative_to(root_dir)
    label = rel_path.parts[0] if len(rel_path.parts) > 1 else "unlabeled"

    try:
        # 1. Metadane bez pełnego dekodowania
        info = sf.info(str(filepath))
        orig_sr = info.samplerate
        channels = info.channels
        subtype = info.subtype
    except Exception:
        orig_sr, channels, subtype = None, None, "unknown"

    try:
        # 2. Wczytanie sygnału w oryginalnym sample rate (mono do analizy widma)
        y, sr = librosa.load(str(filepath), sr=None, mono=True)
        duration_total = librosa.get_duration(y=y, sr=sr)

        if len(y) == 0 or duration_total == 0:
            return {"file": str(rel_path), "label": label, "error": "Empty audio"}

        # 3. Analiza ciszy na początku/końcu (próg 30 dB poniżej szczytu)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        duration_active = librosa.get_duration(y=y_trimmed, sr=sr)
        silence_ratio = 1.0 - (duration_active / duration_total)

        # Jeśli po odcięciu ciszy sygnał jest pusty, analizujemy oryginał
        y_proc = y_trimmed if len(y_trimmed) > sr * 0.1 else y

        # 4. Cechy amplitudowe i dynamika (pomaga dobrać normalizację)
        rms = librosa.feature.rms(y=y_proc)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        peak_amp = float(np.max(np.abs(y_proc)))
        # Crest factor (stosunek wartości szczytowej do RMS) - wysoki przy stukaniu/klikaniu (impulsach)
        crest_factor = float(peak_amp / (rms_mean + 1e-9))
        clipping_ratio = float(np.mean(np.abs(y_proc) >= 0.99))

        # 5. Cechy widmowe (pomagają dobrać sample_rate, fmax oraz n_mels)
        centroid = librosa.feature.spectral_centroid(y=y_proc, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_proc, sr=sr)[0]
        rolloff_85 = librosa.feature.spectral_rolloff(y=y_proc, sr=sr, roll_percent=0.85)[0]
        rolloff_95 = librosa.feature.spectral_rolloff(y=y_proc, sr=sr, roll_percent=0.95)[0]
        flatness = librosa.feature.spectral_flatness(y=y_proc)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y_proc)[0]

        # 6. Cechy transjentów / rytmiczności (kluczowe dla "Click of Death" i doboru hop_length)
        onset_env = librosa.onset.onset_strength(y=y_proc, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_rate_per_sec = float(len(onsets) / max(duration_active, 0.1))
        onset_strength_max = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
        onset_strength_mean = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0

        return {
            "file": str(rel_path),
            "label": label,
            "orig_sr": orig_sr or sr,
            "channels": channels or 1,
            "format_subtype": subtype,
            "duration_total_sec": round(duration_total, 3),
            "duration_active_sec": round(duration_active, 3),
            "silence_ratio": round(silence_ratio, 4),
            "rms_mean": round(rms_mean, 5),
            "rms_std": round(rms_std, 5),
            "peak_amp": round(peak_amp, 4),
            "crest_factor": round(crest_factor, 3),
            "clipping_ratio": round(clipping_ratio, 5),
            "spec_centroid_mean_hz": round(float(np.mean(centroid)), 1),
            "spec_centroid_std_hz": round(float(np.std(centroid)), 1),
            "spec_bandwidth_mean_hz": round(float(np.mean(bandwidth)), 1),
            "spec_rolloff_85_hz": round(float(np.mean(rolloff_85)), 1),
            "spec_rolloff_95_hz": round(float(np.mean(rolloff_95)), 1),
            "spec_flatness_mean": round(float(np.mean(flatness)), 5),
            "zcr_mean": round(float(np.mean(zcr)), 5),
            "onset_rate_per_sec": round(onset_rate_per_sec, 2),
            "onset_strength_mean": round(onset_strength_mean, 3),
            "onset_strength_max": round(onset_strength_max, 3),
            "error": ""
        }

    except Exception as e:
        return {"file": str(rel_path), "label": label, "error": str(e)}

def main():
    root = Path(DATASET_DIR)
    if not root.exists():
        print(f"Błąd: Katalog '{DATASET_DIR}' nie istnieje. Zmień ścieżkę w zmiennej DATASET_DIR.")
        return

    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    print(f"Znaleziono {len(files)} plików audio. Rozpoczynam ekstrakcję statystyk...")

    results = []
    for f in tqdm(files, desc="Analiza plików"):
        results.append(analyze_audio_file(f, root))

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nGotowe! Zapisano raport do pliku: {OUTPUT_CSV}")
    print(f"Prześlij plik '{OUTPUT_CSV}' w czacie, aby dobrać metodę enkodowania.")

if __name__ == "__main__":
    main()