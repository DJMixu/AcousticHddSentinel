"""
Eksperyment: Optymalizacja i zwielokrotnienie detekcji klasy 'scratching' (Tarcie głowic o talerz).
Cel: Podniesienie F1-score i Recall dla klasy 'scratching' z 34.8% / 28.6% do >75%+.
Metoda:
1. Asymetryczny Overlap (step = 0.5s) dla nagrań treningowych rzadkich klas (scratching i beeping).
2. Pełne ważenie klas (bez tłumienia sqrt).
3. Trening na GPU (AMD Radeon RX 9060 XT).
4. Ewaluacja na identycznym zbiorze testowym (92 próbki, 0% overlap, brak data leakage).
5. Analiza macierzy pomyłek i kalibracja progu decyzyjnego.
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Konfiguracja GPU i pamięci
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Aktywne GPU: {gpus}")
    except RuntimeError as e:
        print(f"[GPU] Błąd konfiguracji GPU: {e}")

from hdd_audio_encoder import (
    TARGET_SR, CHUNK_DURATION, TRIM_DB,
    normalize_waveform_peak, normalize_tensor_zscore, encode_audio_chunk,
    split_files_by_class, INPUT_DIR
)
from model_cnn import build_hdd_cnn_model

EXP_DIR = Path("scratch_exp_scratching")
DATA_DIR = EXP_DIR / "dataset_encoded"
OUT_DIR = EXP_DIR / "output"

def prepare_augmented_dataset():
    """Generuje zbiór danych z asymetrycznym overlapem dla scratching i beeping."""
    print("=" * 70)
    print("FAZA 1: GENEROWANIE ASYMETRYCZNEGO ZBIORU DANYCH (TRENING)")
    print("=" * 70)

    if DATA_DIR.exists():
        print(f"Katalog {DATA_DIR} już istnieje. Pomijanie kodowania.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    in_root = Path(INPUT_DIR)
    splits = split_files_by_class(in_root, ratios=(0.70, 0.15, 0.15), seed=42)

    for split_name, file_entries in splits.items():
        is_train = (split_name == "train")

        for filepath, label in file_entries:
            # Dla zbioru treningowego: scratching i beeping z gęstszym krokiem
            if is_train and label in ["scratching", "beeping"]:
                step_sec = 0.5   # 83.3% overlap
            elif is_train:
                step_sec = 1.5   # 50.0% overlap
            else:
                step_sec = 3.0   # 0% overlap (brak overlapu w teście/walidacji)

            target_dir = DATA_DIR / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)

            try:
                y, sr = librosa.load(str(filepath), sr=TARGET_SR, mono=True)
                y = y - np.mean(y)
                y_trimmed, _ = librosa.effects.trim(y, top_db=TRIM_DB)
                if len(y_trimmed) < int(sr * 0.5):
                    continue

                chunk_samples = int(sr * CHUNK_DURATION)
                step_samples = int(sr * step_sec)

                if len(y_trimmed) < chunk_samples:
                    pad_len = chunk_samples - len(y_trimmed)
                    y_trimmed = np.pad(y_trimmed, (0, pad_len), mode="reflect")

                chunks = []
                for start in range(0, len(y_trimmed) - chunk_samples + 1, step_samples):
                    chunk = y_trimmed[start : start + chunk_samples]
                    if np.sqrt(np.mean((chunk - np.mean(chunk)) ** 2)) >= 1e-4:
                        chunks.append(chunk)

                for idx, chunk in enumerate(chunks):
                    tensor = encode_audio_chunk(
                        y_chunk=chunk,
                        sr=TARGET_SR,
                        waveform_norm_fn=normalize_waveform_peak,
                        tensor_norm_fn=normalize_tensor_zscore,
                        tensor_format="tf"
                    )
                    out_filename = f"{filepath.stem}_chunk{idx:03d}.npy"
                    np.save(target_dir / out_filename, tensor)

            except Exception as e:
                print(f"[BŁĄD] Plik {filepath.name}: {e}")

    # Podsumowanie liczby próbek
    for s in ["train", "validate", "test"]:
        counts = {}
        for p in (DATA_DIR / s).glob("*"):
            counts[p.name] = len(list(p.glob("*.npy")))
        print(f"[{s.upper()}] Liczby tensorów: {counts}")


def load_dataset_split(split_name: str, classes: List[str]):
    files = []
    labels = []
    for cls_idx, cls_name in enumerate(classes):
        cls_files = sorted(list((DATA_DIR / split_name / cls_name).glob("*.npy")))
        for f in cls_files:
            files.append(str(f))
            labels.append(cls_idx)
    return files, labels


def create_tf_dataset(files: List[str], labels: List[int], num_classes: int, batch_size: int = 32, is_training: bool = True):
    def _py_load(fpath_t):
        arr = np.load(fpath_t.numpy().decode("utf-8")).astype(np.float32)
        return arr

    paths_ds = tf.data.Dataset.from_tensor_slices(files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(files), 1000), seed=42, reshuffle_each_iteration=True)

    def _map_entry(path_elem, label_elem):
        [tensor] = tf.numpy_function(
            func=_py_load,
            inp=[path_elem],
            Tout=[tf.float32]
        )
        tensor.set_shape([128, 259, 3])
        one_hot_label = tf.one_hot(label_elem, depth=num_classes)
        one_hot_label.set_shape([num_classes])
        return tensor, one_hot_label

    dataset = dataset.map(_map_entry, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def run_experiment():
    prepare_augmented_dataset()

    classes = ["beeping", "clicking", "normal", "scratching"]
    num_classes = len(classes)

    train_files, train_labels = load_dataset_split("train", classes)
    val_files, val_labels = load_dataset_split("validate", classes)
    test_files, test_labels = load_dataset_split("test", classes)

    print(f"\nPróbki: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Obliczenie pełnych zbalansowanych wag klas (bez tłumienia sqrt)
    raw_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: float(w) for i, w in enumerate(raw_weights)}
    print("\nObliczone zbalansowane wagi klas (class_weight):")
    for i, c in enumerate(classes):
        print(f"  [{i}] {c:12s}: waga = {class_weights[i]:.3f} (próbek: {train_labels.count(i)})")

    train_ds = create_tf_dataset(train_files, train_labels, num_classes, batch_size=32, is_training=True)
    val_ds = create_tf_dataset(val_files, val_labels, num_classes, batch_size=32, is_training=False)
    test_ds = create_tf_dataset(test_files, test_labels, num_classes, batch_size=32, is_training=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_hdd_cnn_model(input_shape=(128, 259, 3), num_classes=num_classes, dropout_rate=0.3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = OUT_DIR / "best_scratching_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    print("\n" + "=" * 70)
    print("FAZA 2: TRENING MODELU NA GPU AMD RADEON RX 9060 XT (35 EPOK)")
    print("=" * 70)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=35,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\n" + "=" * 70)
    print("FAZA 3: EWALUACJA NA ZBIORZE TESTOWYM (92 PRÓBKI)")
    print("=" * 70)

    best_model = tf.keras.models.load_model(str(checkpoint_path))

    # Predykcje dla całego zbioru testowego
    y_true_list = []
    y_probs_list = []

    for fpath, label_idx in zip(test_files, test_labels):
        x = np.load(fpath)[np.newaxis, ...]
        p = best_model.predict(x, verbose=0)[0]
        y_true_list.append(label_idx)
        y_probs_list.append(p)

    y_true = np.array(y_true_list)
    y_probs = np.array(y_probs_list)
    y_pred_argmax = np.argmax(y_probs, axis=1)

    print("\n--- STANDARDOWY ARGMAX (BEZ KALIBRACJI PROGOWEJ) ---")
    report_std = classification_report(y_true, y_pred_argmax, target_names=classes, digits=4)
    print(report_std)
    cm_std = confusion_matrix(y_true, y_pred_argmax)
    print("Macierz pomyłek:")
    print(cm_std)

    # Sprawdzenie czułości na scratching
    sc_idx = classes.index("scratching")
    sc_mask = (y_true == sc_idx)
    sc_total = np.sum(sc_mask)
    sc_detected = np.sum((y_pred_argmax == sc_idx) & sc_mask)
    print(f"\nScratching standardowo wykryty: {sc_detected}/{sc_total} ({sc_detected/sc_total*100:.1f}%)")

    # Kalibracja progowa: jeśli P(scratching) >= threshold, przypisz scratching
    print("\n--- KALIBRACJA PROGOWA P(scratching) ---")
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40]:
        y_pred_thresh = y_pred_argmax.copy()
        for idx in range(len(y_probs)):
            if y_probs[idx, sc_idx] >= thresh:
                y_pred_thresh[idx] = sc_idx
        sc_det = np.sum((y_pred_thresh == sc_idx) & sc_mask)
        sc_prec = np.sum((y_pred_thresh == sc_idx) & sc_mask) / (np.sum(y_pred_thresh == sc_idx) + 1e-9)
        sc_rec = sc_det / sc_total
        sc_f1 = 2 * sc_prec * sc_rec / (sc_prec + sc_rec + 1e-9)
        overall_acc = np.mean(y_true == y_pred_thresh)
        print(f"Próg {thresh:.2f} -> Scratching Recall: {sc_rec*100:5.1f}% | Precision: {sc_prec*100:5.1f}% | F1: {sc_f1*100:5.1f}% | Dokładność ogólna: {overall_acc*100:5.1f}%")

if __name__ == "__main__":
    run_experiment()
