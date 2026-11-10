import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Callable
import numpy as np
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. KONFIGURACJA PIPELINE'U
# =====================================================================
INPUT_DIR = "./CNN/dataset_categorized" if os.path.exists("./CNN/dataset_categorized") else "./dataset_categorized"
OUTPUT_DIR = "./dataset_encoded"      # Katalog wyjściowy (utworzy podkatalogi train/, validate/, test/)

# Proporcje podziału plików źródłowych (muszą sumować się do 1.0)
SPLIT_RATIOS = (0.70, 0.15, 0.15)     # (train, validate, test)
RANDOM_SEED = 42

# Format wyjściowy tensorów: "tf" -> (128, 259, 3) dla TensorFlow/Keras | "pytorch" -> (3, 128, 259) dla PyTorch
TENSOR_FORMAT = "tf"

# Parametry akustyczne (dobrane na podstawie analizy pliku audio_dataset_profile.csv)
TARGET_SR = 22050
CHUNK_DURATION = 3.0                  # Długość okna w sekundach
OVERLAP_DURATION = 1.5                # Nakładanie się okien (1.5s = 50% overlap w zbiorze train)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 50.0
FMAX = 10000.0
TRIM_DB = 30.0                        # Próg odcięcia ciszy na początku i końcu nagrania

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg'}


# =====================================================================
# 2. WYIZOLOWANE FUNKCJE NORMALIZACJI (ŁATWE DO IMPORTU / PODMIANY)
# =====================================================================

def normalize_waveform_peak(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalizacja szczytowa (Peak Normalization) surowego sygnału audio do zakresu [-1.0, 1.0].
    Idealna do zachowania wysokiego Crest Factor (ostrych kliknięć głowicy).
    """
    y = y - np.mean(y)  # Usunięcie DC offsetu (dochodzącego do 0.377 w surowych plikach)
    peak = np.max(np.abs(y))
    if peak > eps:
        return y / peak
    return y


def normalize_waveform_rms(y: np.ndarray, target_rms: float = 0.05, eps: float = 1e-6) -> np.ndarray:
    """
    Alternatywna normalizacja energii (RMS Normalization) surowego sygnału audio.
    Wyrównuje średnią głośność tła, z zabezpieczeniem przed przesterowaniem (clippingiem).
    """
    y = y - np.mean(y)
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > eps:
        y_norm = y * (target_rms / current_rms)
        return np.clip(y_norm, -1.0, 1.0)
    return y


def normalize_tensor_zscore(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """
    Standaryzacja Z-score (średnia = 0, odchylenie standardowe = 1) osobno dla każdego kanału
    (Log-Mel, Delta, Delta-Delta). Najlepsza dla szybkiej zbieżności sieci CNN.
    """
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        if channel_axis == 0:
            ch = tensor[c]
            norm_tensor[c] = (ch - np.mean(ch)) / (np.std(ch) + eps)
        else:
            ch = tensor[..., c]
            norm_tensor[..., c] = (ch - np.mean(ch)) / (np.std(ch) + eps)

    return norm_tensor


def normalize_tensor_minmax(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """
    Alternatywna normalizacja Min-Max skalująca każdy kanał spektrogramu do przedziału [0.0, 1.0].
    """
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        if channel_axis == 0:
            ch = tensor[c]
            c_min, c_max = np.min(ch), np.max(ch)
            norm_tensor[c] = (ch - c_min) / (c_max - c_min + eps)
        else:
            ch = tensor[..., c]
            c_min, c_max = np.min(ch), np.max(ch)
            norm_tensor[..., c] = (ch - c_min) / (c_max - c_min + eps)

    return norm_tensor


# =====================================================================
# 3. MODUŁ EKSTRAKCJI CECH I CIĘCIA NA OKNA
# =====================================================================

def encode_audio_chunk(
    y_chunk: np.ndarray,
    sr: int = TARGET_SR,
    waveform_norm_fn: Callable[[np.ndarray], np.ndarray] = normalize_waveform_peak,
    tensor_norm_fn: Callable[[np.ndarray, int], np.ndarray] = normalize_tensor_zscore,
    tensor_format: str = TENSOR_FORMAT
) -> np.ndarray:
    """
    Przekształca pojedyncze okno audio (np. 3 sekundy) w znormalizowany 3-kanałowy tensor.
    Można łatwo wywołać w zewnętrznym skrypcie podczas inferencji (predykcji).
    """
    # Krok 1: Wyizolowana normalizacja fali dźwiękowej
    if waveform_norm_fn is not None:
        y_chunk = waveform_norm_fn(y_chunk)

    # Krok 2: Wyliczenie Log-Mel Spektrogramu
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

    # Krok 3: Wyliczenie pochodnych czasowych (Delta i Delta-Delta)
    delta = librosa.feature.delta(log_mel, width=9, order=1, mode="nearest")
    delta2 = librosa.feature.delta(log_mel, width=9, order=2, mode="nearest")

    # Krok 4: Złożenie kanałów w odpowiednim formacie (TensorFlow vs PyTorch)
    if tensor_format.lower() == "pytorch":
        # Kształt: (3, 128, 259)
        tensor = np.stack([log_mel, delta, delta2], axis=0).astype(np.float32)
        ch_axis = 0
    else:
        # Kształt dla TensorFlow / Keras: (128, 259, 3)
        tensor = np.stack([log_mel, delta, delta2], axis=-1).astype(np.float32)
        ch_axis = -1

    # Krok 5: Wyizolowana normalizacja kanałów tensora
    if tensor_norm_fn is not None:
        tensor = tensor_norm_fn(tensor, ch_axis)

    return tensor


def slice_waveform_into_chunks(y: np.ndarray, sr: int, is_train: bool = True) -> List[np.ndarray]:
    """
    Odcina ciszę na końcach i dzieli sygnał na równe 3-sekundowe fragmenty.
    Dla zbioru 'train' stosuje 50% overlap (więcej próbek), a dla 'validate'/'test' brak nakładania.
    """
    chunk_samples = int(sr * CHUNK_DURATION)
    # W zbiorze walidacyjnym i testowym możemy wyłączyć overlap lub zostawić taki sam krok
    step_samples = int(sr * (CHUNK_DURATION - OVERLAP_DURATION)) if is_train else chunk_samples

    y = y - np.mean(y)
    y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    if len(y_trimmed) < int(sr * 0.5):
        return []

    # Odbicie lustrzane zamiast np.tile zapobiega skokom fazy (sztucznym kliknięciom)
    if len(y_trimmed) < chunk_samples:
        pad_len = chunk_samples - len(y_trimmed)
        y_trimmed = np.pad(y_trimmed, (0, pad_len), mode="reflect")

    chunks = []
    for start in range(0, len(y_trimmed) - chunk_samples + 1, step_samples):
        chunk = y_trimmed[start : start + chunk_samples]
        # Odrzucenie fragmentów całkowicie cichych w środku nagrania
        if np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2)) >= 1e-4:
            chunks.append(chunk)

    return chunks


# =====================================================================
# 4. PODZIAŁ PLIKÓW NA TRAIN / VALIDATE / TEST (BEZ DATA LEAKAGE)
# =====================================================================

def split_files_by_class(
    input_root: Path,
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED
) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Grupuje pliki według klasy (podfolderu) i dzieli całe pliki źródłowe na
    zbiory 'train', 'validate' oraz 'test', zapobiegając wyciekowi danych.
    """
    random.seed(seed)
    files_by_label: Dict[str, List[Path]] = {}

    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            rel_path = p.relative_to(input_root)
            label = rel_path.parts[0] if len(rel_path.parts) > 1 else "unlabeled"
            files_by_label.setdefault(label, []).append(p)

    splits = {"train": [], "validate": [], "test": []}

    for label, file_list in files_by_label.items():
        # Sortowanie przed przetasowaniem gwarantuje 100% powtarzalność (reproducibility)
        file_list = sorted(file_list)
        random.shuffle(file_list)

        n = len(file_list)
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else 0
        
        # Korekta, aby nie przekroczyć liczby plików w małych klasach
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1

        train_files = file_list[:n_train]
        val_files = file_list[n_train : n_train + n_val]
        test_files = file_list[n_train + n_val :]

        splits["train"].extend([(f, label) for f in train_files])
        splits["validate"].extend([(f, label) for f in val_files])
        splits["test"].extend([(f, label) for f in test_files])

    return splits


# =====================================================================
# 5. GŁÓWNA PĘTLA PRZETWARZANIA
# =====================================================================

def build_dataset():
    in_root = Path(INPUT_DIR)
    out_root = Path(OUTPUT_DIR)

    if not in_root.exists():
        print(f"Błąd: Katalog wejściowy '{INPUT_DIR}' nie istnieje.")
        return

    if out_root.exists():
        shutil.rmtree(out_root)

    splits = split_files_by_class(in_root)
    summary_stats = {"train": 0, "validate": 0, "test": 0, "skipped_files": 0}

    print(f"Podział plików źródłowych:")
    print(f"  - TRAIN:    {len(splits['train'])} plików")
    print(f"  - VALIDATE: {len(splits['validate'])} plików")
    print(f"  - TEST:     {len(splits['test'])} plików\n")

    for split_name, file_entries in splits.items():
        is_train = (split_name == "train")

        for filepath, label in tqdm(file_entries, desc=f"Generowanie [{split_name.upper()}]"):
            try:
                y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
                chunks = slice_waveform_into_chunks(y, sr, is_train=is_train)

                if not chunks:
                    summary_stats["skipped_files"] += 1
                    continue

                # Ścieżka docelowa: np. ./dataset_encoded/train/clicking/
                target_dir = out_root / split_name / label
                target_dir.mkdir(parents=True, exist_ok=True)

                for idx, chunk in enumerate(chunks):
                    # Przekazujemy wyizolowane funkcje normalizacji jako argumenty
                    tensor = encode_audio_chunk(
                        y_chunk=chunk,
                        sr=TARGET_SR,
                        waveform_norm_fn=normalize_waveform_peak,
                        tensor_norm_fn=normalize_tensor_zscore,
                        tensor_format=TENSOR_FORMAT
                    )

                    out_filename = f"{filepath.stem}_chunk{idx:03d}.npy"
                    np.save(target_dir / out_filename, tensor)
                    summary_stats[split_name] += 1

            except Exception as e:
                summary_stats["skipped_files"] += 1
                print(f"\n[BŁĄD] Pominięto plik '{filepath.name}': {e}")

    print("\n" + "=" * 55)
    print("PODSUMOWANIE ENKODOWANIA:")
    print(f"  Format tensorów:  {TENSOR_FORMAT.upper()} -> (128, 259, 3)" if TENSOR_FORMAT == "tf" else "  Format tensorów:  PYTORCH -> (3, 128, 259)")
    print(f"  Próbki TRAIN:     {summary_stats['train']} plików .npy")
    print(f"  Próbki VALIDATE:  {summary_stats['validate']} plików .npy")
    print(f"  Próbki TEST:      {summary_stats['test']} plików .npy")
    print(f"  Uszkodzone pliki: {summary_stats['skipped_files']}")
    print("=" * 55)

if __name__ == "__main__":
    build_dataset()
