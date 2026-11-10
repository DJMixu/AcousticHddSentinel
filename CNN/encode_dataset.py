import os
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# === OPTYMALNE PARAMETRY ENKODOWANIA (DOBRANE Z PROFILU CSV) ===
INPUT_DIR = "./NOWY"             # Twój folder z plikami audio (najlepiej podzielony na podfoldery klas)
OUTPUT_DIR = "./encoded_dataset" # Folder wyjściowy na tensory .npy

TARGET_SR = 22050                # 99% Twoich plików ma 22050 Hz
CHUNK_DURATION = 3.0             # 3.0 sekundy (najkrótszy aktywny plik ma 3.88 s)
OVERLAP_DURATION = 1.5           # 50% nakładania się okien (krok co 1.5 s)
N_FFT = 1024                     # Okno ~46.4 ms
HOP_LENGTH = 256                 # Krok ~11.6 ms (zachowuje ostre impulsy przy wysokim Crest Factor = 35-208)
N_MELS = 128                     # Wysokość obrazu spektrogramu
FMIN = 50.0                      # Odcięcie buczenia/drgań poniżej 50 Hz
FMAX = 10000.0                   # Zachowanie wysokich częstotliwości tarcia (spec_rolloff_95 sięga ~9981 Hz)

CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION)
STEP_SAMPLES = int(TARGET_SR * (CHUNK_DURATION - OVERLAP_DURATION))
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}

def extract_3channel_features(y_chunk: np.ndarray, sr: int) -> np.ndarray:
    """
    Przekształca 3-sekundowy wycinek audio w 3-kanałowy tensor (3, 128, 259):
    Kanał 0: Log-Mel Spectrogram
    Kanał 1: Delta (1. pochodna czasowa - detekcja kliknięć/transjentów)
    Kanał 2: Delta-Delta (2. pochodna czasowa - dynamika zmian/spin-down)
    """
    # 1. Peak-normalization wycinka (kluczowe przy rozrzucie RMS od 0.0006 do 0.378)
    max_amp = np.max(np.abs(y_chunk))
    if max_amp > 1e-6:
        y_chunk = y_chunk / max_amp

    # 2. Mel-spektrogram mocy -> skala decybelowa (Log-Mel)
    mel_spec = librosa.feature.melspectrogram(
        y=y_chunk,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # 3. Cechy dynamiczne (Delta i Delta-Delta)
    delta = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)

    # 4. Złożenie w tensor 3D: (3, n_mels, time_frames) -> (3, 128, 259)
    tensor = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)

    # 5. Standaryzacja per-channel (Instance Normalization: mean=0, std=1)
    for c in range(tensor.shape[0]):
        mean = np.mean(tensor[c])
        std = np.std(tensor[c])
        tensor[c] = (tensor[c] - mean) / (std + 1e-6)

    return tensor

def process_dataset():
    in_root = Path(INPUT_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    total_chunks = 0
    skipped_files = 0

    for filepath in tqdm(files, desc="Enkodowanie plików"):
        rel_path = filepath.relative_to(in_root)
        label = rel_path.parts[0] if len(rel_path.parts) > 1 else "unlabeled"
        
        try:
            # Wczytanie i automatyczny resampling (np. dla Audio_Player_57.wav z 44.1kHz stereo)
            y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
            
            # Odcięcie ciszy na brzegach (próg 30 dB)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            if len(y_trimmed) < TARGET_SR * 0.5:
                skipped_files += 1
                continue

            # Jeśli nagranie po odcięciu ciszy jest krótsze niż 3.0s, powielamy je cyklicznie (tile)
            if len(y_trimmed) < CHUNK_SAMPLES:
                repeats = int(np.ceil(CHUNK_SAMPLES / len(y_trimmed)))
                y_trimmed = np.tile(y_trimmed, repeats)[:CHUNK_SAMPLES]

            # Tworzenie katalogu docelowego dla danej klasy
            class_out_dir = out_root / label
            class_out_dir.mkdir(parents=True, exist_ok=True)

            # Cięcie na 3-sekundowe okna z przesunięciem 1.5s
            chunk_idx = 0
            for start in range(0, len(y_trimmed) - CHUNK_SAMPLES + 1, STEP_SAMPLES):
                chunk = y_trimmed[start : start + CHUNK_SAMPLES]
                
                # Pominięcie pustych okien (jeśli w środku nagrania była długa przerwa)
                if np.sqrt(np.mean(chunk**2)) < 1e-4:
                    continue

                tensor = extract_3channel_features(chunk, TARGET_SR)
                
                out_name = f"{filepath.stem}_chunk{chunk_idx:03d}.npy"
                np.save(class_out_dir / out_name, tensor)
                chunk_idx += 1
                total_chunks += 1

        except Exception as e:
            skipped_files += 1
            print(f"\nPominięto uszkodzony plik {filepath.name}: {e}")

    print(f"\nZakończono! Wygenerowano {total_chunks} tensorów o wymiarach (3, 128, 259).")
    print(f"Pominięto plików: {skipped_files}")

if __name__ == "__main__":
    process_dataset()