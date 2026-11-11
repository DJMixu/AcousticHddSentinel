"""
HDD Acoustic Fault Detection - Audio to .npy Tensor Encoder
Single Source of Truth: ADR-001 & AGENTS.md

Enkoder surowych plików audio do 3-kanałowych reprezentacji czasowo-częstotliwościowych:
- Kanał 0: Log-Mel Spektrogram (struktura harmoniczna, piski, rezonanse)
- Kanał 1: Delta – 1. pochodna czasowa (transjenty, impulsowe klikanie głowic)
- Kanał 2: Delta-Delta – 2. pochodna czasowa (dynamika przyspieszeń i zmian kątowych)

Zgodność z ADR-001:
1. Podział na poziomie całych nagrań źródłowych (Zero Data Leakage).
2. Kształt tensora na dysku: (128, 259, 3) float32 (Channel-Last).
3. Parametry DSP: SR=22050, N_FFT=1024, HOP_LENGTH=256, N_MELS=128, FMIN=50.0, FMAX=10000.0.
4. Normalizacja dwustopniowa: Peak Normalization fali + Z-score per kanał tensora.
5. Filtracja martwych ramek i szumu tła (RMS >= 0.002).
6. Zbalansowana augmentacja w zbiorze train (pitch-shift +/-1.5 półtonu dla klas rzadkich).
7. Zbiory validate i test w 100% czyste, bez nakładania okien (step=3.0s).
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Optional
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# 1. KONFIGURACJA PIPELINE'U
# =====================================================================
INPUT_DIR = Path("dataset_categorized")
OUTPUT_DIR = Path("dataset_encoded")

# Proporcje podziału plików źródłowych (Zero Data Leakage)
SPLIT_RATIOS = (0.68, 0.14, 0.18)     # (train, validate, test)
RANDOM_SEED = 42

TARGET_SR = 22050
CHUNK_DURATION = 3.0                  # Długość okna w sekundach
OVERLAP_DURATION = 1.5                # Nakładanie się okien w train (50% overlap)
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 50.0
FMAX = 10000.0
TRIM_DB = 30.0

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


# =====================================================================
# 2. WYIZOLOWANE FUNKCJE NORMALIZACJI
# =====================================================================

def normalize_waveform_peak(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalizacja szczytowa (Peak Normalization) fali audio do zakresu [-1.0, 1.0]."""
    y = y - np.mean(y)  # Usunięcie DC offsetu
    peak = np.max(np.abs(y))
    if peak > eps:
        return y / peak
    return y


def normalize_waveform_rms(y: np.ndarray, target_rms: float = 0.05, eps: float = 1e-6) -> np.ndarray:
    """Alternatywna normalizacja energii (RMS) surowego sygnału audio."""
    y = y - np.mean(y)
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > eps:
        y_norm = y * (target_rms / current_rms)
        return np.clip(y_norm, -1.0, 1.0)
    return y


def normalize_tensor_zscore(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """Standaryzacja Z-score osobno dla każdego kanału (Log-Mel, Delta, Delta-Delta)."""
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        channel_slice = np.take(tensor, indices=c, axis=channel_axis)
        mean = np.mean(channel_slice)
        std = np.std(channel_slice)
        if std < eps:
            std = eps
        normalized_channel = (channel_slice - mean) / std
        if channel_axis == -1 or channel_axis == tensor.ndim - 1:
            norm_tensor[..., c] = normalized_channel
        else:
            norm_tensor[c, ...] = normalized_channel

    return norm_tensor


def normalize_tensor_minmax(tensor: np.ndarray, channel_axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """Alternatywna normalizacja Min-Max skalująca każdy kanał do [0.0, 1.0]."""
    norm_tensor = np.empty_like(tensor, dtype=np.float32)
    num_channels = tensor.shape[channel_axis]

    for c in range(num_channels):
        channel_slice = np.take(tensor, indices=c, axis=channel_axis)
        c_min = np.min(channel_slice)
        c_max = np.max(channel_slice)
        denom = c_max - c_min
        if denom < eps:
            denom = eps
        normalized_channel = (channel_slice - c_min) / denom
        if channel_axis == -1 or channel_axis == tensor.ndim - 1:
            norm_tensor[..., c] = normalized_channel
        else:
            norm_tensor[c, ...] = normalized_channel

    return norm_tensor


# =====================================================================
# 3. EKSTRAKCJA CECH CZASOWO-CZĘSTOTLIWOŚCIOWYCH
# =====================================================================

def encode_audio_chunk(
    y_chunk: np.ndarray,
    sr: int = TARGET_SR,
    waveform_norm_fn: Callable[[np.ndarray], np.ndarray] = normalize_waveform_peak,
    tensor_norm_fn: Callable[[np.ndarray], np.ndarray] = normalize_tensor_zscore,
    tensor_format: str = "tf"
) -> np.ndarray:
    """
    Konwertuje pojedynczy wycinek audio 3.0s do 3-kanałowego tensora [Log-Mel, Delta, Delta-Delta].
    Zwraca tensor float32 o stałym kształcie:
    - (128, 259, 3) dla tensor_format="tf" (Channel-Last, format na dysku .npy)
    - (3, 128, 259) dla tensor_format="pytorch" (Channel-First)
    """
    target_samples = int(CHUNK_DURATION * sr)  # 66 150 próbek
    if len(y_chunk) < target_samples:
        pad_width = target_samples - len(y_chunk)
        y_chunk = np.pad(y_chunk, (0, pad_width), mode="constant")
    elif len(y_chunk) > target_samples:
        y_chunk = y_chunk[:target_samples]

    if waveform_norm_fn is not None:
        y_chunk = waveform_norm_fn(y_chunk)

    # 1. Spektrogram Mel
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

    # 2. Skala logarytmiczna (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Wymuszenie stałej liczby ramek czasowych: 259
    target_frames = 259
    if log_mel.shape[1] < target_frames:
        pad_f = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_f)), mode="edge")
    elif log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]

    # 3. Kanały pochodnych (Delta i Delta-Delta)
    delta_1 = librosa.feature.delta(log_mel, order=1)
    delta_2 = librosa.feature.delta(log_mel, order=2)

    if tensor_format == "pytorch":
        raw_tensor = np.stack([log_mel, delta_1, delta_2], axis=0).astype(np.float32)
        norm_tensor = tensor_norm_fn(raw_tensor, channel_axis=0) if tensor_norm_fn is not None else raw_tensor
    else:
        raw_tensor = np.stack([log_mel, delta_1, delta_2], axis=-1).astype(np.float32)
        norm_tensor = tensor_norm_fn(raw_tensor, channel_axis=-1) if tensor_norm_fn is not None else raw_tensor

    return norm_tensor.astype(np.float32)


def slice_waveform(
    y: np.ndarray,
    sr: int = TARGET_SR,
    chunk_dur: float = CHUNK_DURATION,
    overlap_dur: float = OVERLAP_DURATION,
    is_train: bool = True
) -> List[np.ndarray]:
    """
    Dzieli sygnał na 3-sekundowe okna.
    - Dla zbioru train: 50% overlap (krok 1.5s).
    - Dla zbiorów validate/test: 0% overlap (krok 3.0s).
    - Filtruje martwe fragmenty o znikomej energii RMS < 0.002.
    """
    step_dur = (chunk_dur - overlap_dur) if (is_train and overlap_dur > 0) else chunk_dur
    chunk_samples = int(chunk_dur * sr)
    step_samples = int(step_dur * sr)

    if len(y) < chunk_samples:
        rms = np.sqrt(np.mean(y ** 2))
        return [y] if rms >= 0.002 else []

    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, step_samples):
        chunk = y[start:start + chunk_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms >= 0.002:  # Odrzucanie szumu mikrofonu i ciszy
            chunks.append(chunk)

    if not chunks and len(y) >= chunk_samples:
        chunks.append(y[:chunk_samples])

    return chunks


def split_files(input_dir: Path, ratios: Tuple[float, float, float] = SPLIT_RATIOS, seed: int = RANDOM_SEED) -> Dict[str, List[Tuple[Path, str]]]:
    """Dzieli zbiór na poziomie całych plików źródłowych (Zero Data Leakage)."""
    random.seed(seed)
    np.random.seed(seed)

    classes = [d.name for d in input_dir.iterdir() if d.is_dir()]
    classes.sort()

    splits: Dict[str, List[Tuple[Path, str]]] = {"train": [], "validate": [], "test": []}

    for cls in classes:
        cls_dir = input_dir / cls
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
        files.sort()
        random.shuffle(files)

        n = len(files)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        # Gwarancja reprezentacji rzadkich klas w zbiorze testowym
        if n >= 3 and (n - (n_train + n_val)) == 0:
            if n_train > 1:
                n_train -= 1

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f in train_files:
            splits["train"].append((f, cls))
        for f in val_files:
            splits["validate"].append((f, cls))
        for f in test_files:
            splits["test"].append((f, cls))

    return splits


def process_dataset(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    ratios: Tuple[float, float, float] = SPLIT_RATIOS
):
    """Enkoduje zbiór danych do plików .npy."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    splits = split_files(input_dir, ratios)

    print("=== PODZIAŁ PLIKÓW ŹRÓDŁOWYCH (ZERO DATA LEAKAGE) ===")
    for split_name, flist in splits.items():
        by_cls = {}
        for _, lbl in flist:
            by_cls[lbl] = by_cls.get(lbl, 0) + 1
        print(f"  {split_name:8s} ({len(flist):2d} plików): {by_cls}")

    chunk_counts = {k: {} for k in splits.keys()}

    for split_name, file_tuples in splits.items():
        is_train = (split_name == "train")
        for fpath, label in tqdm(file_tuples, desc=f"Generowanie [{split_name.upper()}]"):
            out_class_dir = output_dir / split_name / label
            out_class_dir.mkdir(parents=True, exist_ok=True)

            try:
                y, sr = librosa.load(str(fpath), sr=TARGET_SR, mono=True)
            except Exception as e:
                print(f"Błąd odczytu {fpath.name}: {e}")
                continue

            audio_variants = [y]

            # Augmentacja audio WYŁĄCZNIE w zbiorze treningowym dla klas mniejszościowych
            if is_train and label in ["scratching", "beeping"]:
                try:
                    y_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
                    y_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5)
                    audio_variants.extend([y_up, y_down])
                except Exception:
                    pass
            elif is_train and label == "normal":
                try:
                    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.0)
                    audio_variants.append(y_shift)
                except Exception:
                    pass

            for v_idx, y_var in enumerate(audio_variants):
                var_tag = f"_v{v_idx}" if v_idx > 0 else ""
                chunks = slice_waveform(y_var, sr, is_train=is_train)

                for c_idx, chunk in enumerate(chunks):
                    tensor = encode_audio_chunk(chunk, sr=sr, tensor_format="tf")
                    out_fname = f"{fpath.stem}{var_tag}_chunk{c_idx:03d}.npy"
                    np.save(out_class_dir / out_fname, tensor)
                    chunk_counts[split_name][label] = chunk_counts[split_name].get(label, 0) + 1

    print("\n=== PODSUMOWANIE WYGENEROWANYCH TENSORÓW .NPY ===")
    for split_name, counts in chunk_counts.items():
        total = sum(counts.values())
        print(f"  Split: {split_name:8s} -> {total:4d} tensorów: {counts}")


if __name__ == "__main__":
    process_dataset()
